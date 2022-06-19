from turtle import update
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.utils.data as data
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate, layer
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from typing import Callable, overload
import math
import torch.nn.init as init
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import os


# 示例命令
# python main.py --dataset-dir ./data/MNIST/ --model-output-dir ./result/

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cpu', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=1, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=350, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('-N', '--epoch', default=10, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')
parser.add_argument('--eval-mode', default=0, type=int, help='Whether to evaluate or train.')



class new_LIFNode(neuron.LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False, 
                 v_rest: float=-65., E_exc : float = 0., E_inh : float = -100.,
                 hidden_size = 400, device='cpu'):
        assert isinstance(tau, float) and tau > 1.
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        v_init_val = v_rest
        self.v = torch.tensor(np.ones(shape=(hidden_size,))*v_init_val).to(device)
        self.E_exc = E_exc
        self.E_inh = E_inh

    def neuronal_charge(self, ge_averaged, gi_average):
        self.v = self.v + ((self.v_rest - self.v) + ge_averaged * (self.E_exc - self.v) + gi_average * (self.E_inh - self.v)) / self.tau 

    def neuronal_fire(self):
        return (self.v > self.v_threshold).double()

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None: # soft reset
            self.v = self.v - spike_d * self.v_threshold
        else: # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def forward(self, ge_averaged, gi_averaged):
        self.neuronal_charge(ge_averaged, gi_averaged)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def reset(self):
        self.v[:] = self.v_rest

# 突触记录连接权重（使用STDP_layer更新权重）,区分为激发性突触和抑制性突触
# 输入：突触前膜的脉冲， 输出：激发/抑制，突触权重，电流x
class Synapse(nn.Module):
    def __init__(self, in_features = 400, out_features = 400, pre_type = 'e', lr = 1e-2, device = 'cpu'):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias = False).to(device)
        self.ge = torch.tensor(np.ones(shape=(out_features, in_features))*0.1).to(device)
        self.gi = torch.tensor(np.ones(shape=(out_features, in_features))*0.1).to(device)
        self.stdp_learner = layer.STDPLearner(100., 100., self.f_pre, self.f_post)
        self.lr = lr
        self.pre_type = pre_type
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight.weight, a=0.1, b=1.0)

    def forward(self, input_spike, t):
        mask = input_spike.unsqueeze(0)
        if self.pre_type == "e":
            self.ge = mask*(self.ge+self.weight.weight)+(1.0-mask)*self.ge*(1.0-1.0/1)
        else:
            self.gi = mask*(self.gi+self.weight.weight)+(1.0-mask)*self.gi*(1.0-1.0/2)
        ge_averaged = torch.matmul(self.ge, input_spike)
        gi_averaged = torch.matmul(self.gi, input_spike)
        return ge_averaged, gi_averaged

    def f_pre(self, x):
        return x.abs() + 0.1

    def f_post(self, x):
        return - self.f_pre(x)

    @torch.no_grad()
    def norm_weight(self):
        mean, std = torch.mean(self.weight.weight), torch.std(self.weight.weight)
        self.weight.weight -= mean
        self.weight.weight /= std
        self.weight.weight -= torch.min(self.weight.weight)

    def update(self, spike_pre, spike_post):
        self.stdp_learner.stdp(spike_pre, spike_post, self.weight, self.lr)
        self.norm_weight()

    def reset(self):
        self.ge[:,:] = 0.1
        self.gi[:,:] = 0.1

# @torch.no_grad()
#     def stdp(self, s_pre: Tensor, s_post: Tensor, module: nn.Module, learning_rate: float):
#         if isinstance(module, nn.Linear):
#             # update trace
#             self.trace_pre += - self.trace_pre / self.tau_pre + s_pre
#             self.trace_post += - self.trace_post / self.tau_post + s_post

#             # update weight
#             delta_w_pre = self.f_pre(module.weight) * s_pre
#             delta_w_post = self.f_post(module.weight) * s_post.unsqueeze(1)
#             module.weight += (delta_w_pre + delta_w_post) * learning_rate


class Net(nn.Module):
    def __init__(self, hidden=400, lr=0.02, device='cpu'):
        super().__init__()
        self.excit_node = new_LIFNode(tau=100.*1000, v_threshold=-52., v_reset=-65., 
                            v_rest=-65, E_exc=0, E_inh=-100., hidden_size = hidden, device=device)
        self.inhib_node = new_LIFNode(tau=10.*1000, v_threshold=-40., v_reset=-45., 
                            v_rest=-60., E_exc=0, E_inh=-85., hidden_size = hidden, device=device)
        self.synapse_1 = Synapse(784, hidden, pre_type='e', lr=lr, device = device)
        self.synapse_2 = Synapse(hidden, hidden, pre_type='e', lr=lr, device = device)
        self.synapse_3 = Synapse(hidden, hidden, pre_type='i', lr=lr, device = device)

    @torch.no_grad()
    def forward(self, x, t):
        self.spike_1 = x
        self.spike_2 = self.excit_node(*self.synapse_1(self.spike_1, t))
        self.spike_3 = self.inhib_node(*self.synapse_2(self.spike_2, t))
        self.spike_4 = self.excit_node(*self.synapse_3(self.spike_3, t))
        return self.spike_2
    
    def update(self):
        self.synapse_1.update(self.spike_1, self.spike_2)
        self.synapse_2.update(self.spike_2, self.spike_3)
        self.synapse_3.update(self.spike_3, self.spike_4)
        return 

    def reset(self):
        self.synapse_1.reset()
        self.synapse_2.reset()
        self.synapse_3.reset()
        self.excit_node.reset()
        self.inhib_node.reset()

# 得到每个神经元对应的类
def get_new_assignments(result_monitor, input_numbers):
    n_e = 400
    assignments = torch.zeros(n_e)
    input_nums = input_numbers
    maximum_rate = torch.zeros(n_e)
    rate = torch.zeros(n_e)
    for j in range(10):
        # input中各个类有多少个sample
        num_assignments = len(torch.where(input_nums == j)[0])
        print(num_assignments, input_nums == j)
        if num_assignments > 0:
            # 计算对应这个类的平均每个sample的发放频率
            rate = torch.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        print('rate:',rate)
        print('maximum_rate:',rate)
        for i in range(n_e):
            # 对于每个神经元，计算其发放频率最大的类，作为其assignment
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

# assignments是一个[1,400]的数组，它代表这400个神经元各自assigned到的类是哪个
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = torch.zeros(10)
    num_assignments = torch.zeros(10)
    for i in range(10):
        num_assignments[i] = len(torch.where(assignments == i)[0])
        if num_assignments[i] > 0:
            # 计算某个类的所有神经元的脉冲发放频率
            summed_rates[i] = torch.sum(spike_rates[assignments == i]) / num_assignments[i]
    # 获得每个类脉冲发放频率的排序 summed_rates[result[9]]代表最高类的发放频率，summed_rates[results[0]]代表最低类的发放频率,[::-1]代表取倒序
    return torch.argsort(summed_rates, descending=True)

# 根据目前的outputNumbers得到目前的表现，每一个update_interval在performance里面记录一次
def get_current_performance(outputNumbers, input_numbers):
    # current_evaluation = int(current_example_num/update_interval)
    # start_num = current_example_num - update_interval
    # end_num = current_example_num
    difference = outputNumbers[:, 0] - input_numbers[:]
    correct = len(np.where(difference == 0)[0])
    performance = correct / len(outputNumbers) * 100
    return performance

# def get_current_performance(performance, outputNumbers, input_numbers, current_example_num, update_interval):
#     current_evaluation = int(current_example_num/update_interval)
#     start_num = current_example_num - update_interval
#     end_num = current_example_num
#     difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
#     correct = len(np.where(difference == 0)[0])
#     performance[current_evaluation] = correct / float(update_interval) * 100
#     return performance

def save_param(model, assignments, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    state_dict = model.state_dict()
    state_dict['assignments'] = assignments
    torch.save(state_dict, os.path.join(path, name))
    print('Saved parameters to', os.path.join(path, name))

def load_param(model, path, name):
    state_dict = torch.load(os.path.join(path, name))
    assignments = state_dict['assignments'].clone()
    del state_dict['assignments']
    print('Loaded params:')
    print(state_dict)
    print('Loaded assignments:')
    print(assignments)
    model.load_state_dict(state_dict)
    return assignments


def train(args):
    # 根据arg初始化变量
    device = args.device
    dataset_dir = args.dataset_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    train_epoch = args.epoch
    param_save_dir = './trained_params/'
    param_save_name = 'params'

     # 初始化数据加载器，需要重新下载数据可以让download=True
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # 定义并初始化网络
    model = Net(hidden=400, lr=lr, device=device)
    model.to(device)

    # 使用泊松编码
    encoder = encoding.PoissonEncoder()

    for epoch in range(train_epoch):
        # 开始训练
        print("Epoch {}:".format(epoch))
        print("Training...")
        j = 0    # j 记录training sample的数量
        update_interval = 100
        result_monitor = torch.zeros((update_interval, 400)).to(device)   # 记录spike count,只需要记录一个update_interval的即可
        input_numbers = torch.zeros(60000).to(device)   # 记录输入的数字
        outputNumbers = torch.zeros((60000, 10)).to(device)  # 记录输出的结果
        assignments = torch.zeros(400).to(device)
        # performance = np.zeros(6)  # 60000 / 10000 = 6
        
        for img, label in tqdm(train_data_loader):
            # 获取图片和对应的one-hot标签
            img = img.to(device)
            label = label.to(device)

            # 仿真一张图片
            current_spike_count = torch.zeros(400).to(device) # 记录累计的脉冲数量
            for t in range(T):
                encoded_img = encoder(img).double()
                output = model(encoded_img.reshape((784,)), t=t)
                current_spike_count += output
                model.update()
            # print('spike:',current_spike_count)
            # print('weight:',model.synapse_1.weight.weight)
            model.reset()

            # 每隔一个update_interval，更新assignments
            if j % update_interval == 0 and j > 0:
                # performance = get_current_performance(performance, outputNumbers, input_numbers, j, update_interval)
                # print(performance)   # 每10000个sample记录一下performance
                assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
                print(assignments)
            
            # 得到目前的spike count，训练标签，输出结果
            result_monitor[j % update_interval,:] = current_spike_count
            input_numbers[j] = label
            outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval,:]) # 得到预测结果

            y_pred = outputNumbers[j,0]
            print('label:', label[0], 'y_pred:', y_pred)

            # 每个1000张图片保存一次参数
            if j % 1000 == 0:
                save_param(model, assignments, param_save_dir, param_save_name+'_%d_%d'%(epoch,j))

            j += 1
        # 每个epoch输出一次performance
        performance = get_current_performance(outputNumbers, input_numbers)
        print(performance)   # 每10000个sample记录一下performance

def test(args):
    # 根据arg初始化变量
    device = args.device
    dataset_dir = args.dataset_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    param_save_dir = './trained_params/'
    param_save_name = 'params'

     # 初始化数据加载器，需要重新下载数据可以让download=True
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 定义并初始化网络
    model = Net(hidden=400, lr=lr, device=device)
    model.to(device)

    # 使用泊松编码
    encoder = encoding.PoissonEncoder()

    load_epoch = 0
    load_iter = 0

    # 获得训练好的assignments
    trained_assignments = load_param(model, param_save_dir, param_save_name+'_%d_%d'%(load_epoch,load_iter))
    result_monitor = torch.zeros((10000, 400)).to(device)   # 记录spike count
    input_numbers = torch.zeros(10000).to(device)   # 记录输入的数字
    outputNumbers = torch.zeros((10000, 10)).to(device)  # 记录输出的结果
    print("Testing...")
    j = 0
    for img, label in tqdm(test_data_loader):
        # 获取图片和对应的one-hot标签
        img = img.to(device)
        label = label.to(device)

        # 仿真一张图片
        current_spike_count = torch.zeros(400).to(device) # 记录累计的脉冲数量
        for t in range(T):
            encoded_img = encoder(img).double()
            output = model(encoded_img.reshape((784,)), t=t)
            current_spike_count += output
        
        # 得到目前的spike count，训练标签，输出结果
        result_monitor[j,:] = current_spike_count
        input_numbers[j] = label
        outputNumbers[j,:] = get_recognized_number_ranking(trained_assignments, result_monitor[j,:]) # 得到预测结果

        y_pred = outputNumbers[j,0]
        print('label:', label[0], 'y_pred:', y_pred)
        
        j += 1
    
    # 输出performance
    performance = get_current_performance(outputNumbers, input_numbers)
    print(performance)

if __name__ == '__main__':
    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    if args.eval_mode:
        test(args)
    else:
        train(args)