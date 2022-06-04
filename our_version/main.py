import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven import layer
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from typing import Callable, overload
import math
import torch.nn.init as init

class new_LIFNode(neuron.LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False, 
                 v_rest: float=-65., E_exc : float = 0., E_inh : float = -100.,
                 hidden_size = 400, device='cpu'):
        assert isinstance(tau, float) and tau > 1.
        # assert isinstance(tau, float)
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        v_init_val = 1.
        self.v = torch.tensor(np.ones(shape=(hidden_size,), dtype=float)*v_init_val).to(device)
        self.E_exc = E_exc
        self.E_inh = E_inh

    def neuronal_charge(self, ge_averaged, gi_average):
        # 充电公式，见文中公式(1)，与源代码中neuron_eqs_e, neuron_eqs_i
        self.v = self.v + ((self.v_rest - self.v) + ge_averaged * (self.E_exc - self.v) + gi_average * (self.E_inh - self.v)) / self.tau 

    def neuronal_fire(self):
        return (self.v > self.v_threshold).float()
        # return self.surrogate_function(self.v - self.v_threshold)

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

    def forward(self, ge_averaged, gi_averaged):
        self.neuronal_charge(ge_averaged, gi_averaged)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

# 突触记录连接权重（使用STDP_layer更新权重）,区分为激发性突触和抑制性突触
# 输入：突触前膜的脉冲， 输出：激发/抑制，突触权重，电流x
class Synapse(nn.Module):
    def __init__(self, in_features = 400, out_features = 400, lr = 1e-2, device = 'cpu'):
        super().__init__()
        self.module = nn.Linear(in_features, out_features, bias = False).to(device)
        self.ge = nn.Linear(in_features, out_features, bias = False).to(device)
        self.gi = nn.Linear(in_features, out_features, bias = False).to(device)
        self.stdp_learner = layer.STDPLearner(100., 100., self.f_pre, self.f_post)
        self.lr = lr

    def forward(self, input_spike, t):
        # # update g_e, g_i
        # self.ge = 
        # self.gi = 
        ge_averaged = self.ge(input_spike)
        gi_averaged = self.gi(input_spike)
        return ge_averaged, gi_averaged

    def f_pre(self, x):
        return x.abs() + 0.1

    def f_post(self, x):
        return - self.f_pre(x)

    def update(self, spike_pre, spike_post):
        self.stdp_learner.stdp(spike_pre, spike_post, self.module, self.lr)


class Net(nn.Module):
    def __init__(self, lr, device='cpu'):
        super().__init__()
        self.input_node = new_LIFNode(hidden_size = 784, device=device)
        self.excit_node = new_LIFNode(hidden_size = 400, device=device)
        self.inhib_node = new_LIFNode(hidden_size = 400, device=device)
        self.synapse_1 = Synapse(784, 400, lr)
        self.synapse_2 = Synapse(400, 400, lr)
        self.synapse_3 = Synapse(400, 400, lr)

    def forward(self, x, t):
        self.spike_1 = self.input_node(x)
        self.spike_2 = self.excit_node(self.synapse_1(self.spike_1, t))
        self.spike_3 = self.inhib_node(self.synapse_2(self.spike_2, t))
        self.spike_4 = self.excit_node(self.synapse_3(self.spike_3, t))
        return self.spike_2
    
    def update(self):
        self.synapse_1.update(self.spike_1, self.spike_2)
        self.synapse_2.update(self.spike_2, self.spike_3)
        self.synapse_3.update(self.spike_3, self.spike_4)
        return 



# 基本单位: ms, mV
neuron_e = new_LIFNode(tau=100., v_threshold=-52., v_reset=-65., v_rest=-65, E_exc=0, E_inh=-100.)
neuron_i = new_LIFNode(tau=10., v_threshold=-40., v_reset=-45., v_rest=-60., E_exc=0, E_inh=-85.)

tau_e = 1
tau_i = 2
ge = 1
gi = 1
neuron_e.reset()
x = torch.as_tensor([2.])
T = 150
s_list = []
v_list = []

for t in range(T):
    # ge and gi 随时间衰减
    ge = ge*(1-1/tau_e)
    gi = gi*(1-1/tau_i)

    s_list.append(neuron_e(x, ge, gi))
    v_list.append(neuron_e.v)


