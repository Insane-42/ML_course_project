import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly.clock_driven import surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from typing import Callable, overload

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

    def neuronal_charge(self, x: torch.Tensor):
        # 充电公式，见文中公式(1)，与源代码中neuron_eqs_e, neuron_eqs_i
        self.v = self.v + ((self.v_rest - self.v) + self.ge * (self.E_exc - self.v) + self.gi * (self.E_inh - self.v) + x) / self.tau 

    def neuronal_fire(self):

        return self.surrogate_function(self.v - self.v_threshold)

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

    def forward(self, x: torch.Tensor, ge, gi):
        # adjust ge and gi
        self.ge = ge
        self.gi = gi

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

# 突触记录连接权重（使用STDP_layer更新权重）,区分为激发性突触和抑制性突触
# 输入：突触前膜的脉冲， 输出：激发/抑制，突触权重，电流x
class Synapse(nn.Module):
     


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


