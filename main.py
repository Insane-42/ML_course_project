import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from typing import Callable, overload
import argparse
from tqdm import tqdm
import torch.nn.functional as F

# 示例命令
# python main.py --dataset-dir ./data/MNIST/ --model-output-dir ./result/

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cpu', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')


class new_LIFNode(neuron.LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False, 
                 v_rest: float=-65., E_exc : float = 0., E_inh : float = -100.):
        assert isinstance(tau, float) and tau > 1.
        # assert isinstance(tau, float)
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        # self.ge = ge
        # self.gi = gi
        self.E_exc = E_exc
        self.E_inh = E_inh

        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference

    def neuronal_charge(ge, gi):
        # 充电公式，见文中公式(1)，与源代码中neuron_eqs_e, neuron_eqs_i
        self.v = self.v + ((self.v_rest - self.v) + ge * (self.E_exc - self.v) + gi * (self.E_inh - self.v) + x) / self.tau 

    def neuronal_fire(self):
        return (self.v > self.v_threshold).float()

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def forward(ge, gi):
        # adjust ge and gi
        self.ge = ge
        self.gi = gi

        self.neuronal_charge(ge, gi)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

# 突触记录连接权重（使用STDP_layer更新权重）,区分为激发性突触和抑制性突触
# 输入：突触前膜的脉冲， 输出：激发/抑制，突触权重，电流x
# class Synapse(nn.Module):
     


# # 基本单位: ms, mV，激发性和抑制性神经元
# neuron_e = new_LIFNode(tau=100., v_threshold=-52., v_reset=-65., v_rest=-65, E_exc=0, E_inh=-100.)
# neuron_i = new_LIFNode(tau=10., v_threshold=-40., v_reset=-45., v_rest=-60., E_exc=0, E_inh=-85.)

# ge和gi衰减速率
tau_e = 1
tau_i = 2
ge = 1
gi = 1
# 时间相关，基本单位:ms
single_example_time = 350  # 跑一个example的时间
resting_time = 150         # 重置电位所需时间


neuron_e.reset()
x = torch.as_tensor([2.])
T = 150
s_list = []
v_list = []

for t in range(T):
    # ge and gi 随时间衰减
    ge = ge*(1-1/tau_e)
    gi = gi*(1-1/tau_i)

    s_list.append(neuron_e(ge, gi))
    v_list.append(neuron_e.v)


if __name__ == '__main__':
    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    # 根据arg初始化变量
    device = args.device
    dataset_dir = args.dataset_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    train_epoch = args.epoch

     # 初始化数据加载器，需要重新下载数据可以让download=True
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 定义并初始化网络，待完成

    # 使用泊松编码
    encoder = encoding.PoissonEncoder()

    for epoch in range(train_epoch):
        # 开始训练
        print("Epoch {}:".format(epoch))
        print("Training...")
        for img, label in tqdm(train_data_loader):
            # 获取图片和对应的one-hot标签
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            print(encoder(img).float)
            # print(label)

        # 开始测试
        print("Testing...")
