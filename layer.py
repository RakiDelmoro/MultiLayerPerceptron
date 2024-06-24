import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import ReLU, Softmax

def linear_layer(in_features: int, out_features: int, device: str, activation_function: str):
    if activation_function == 'relu':
        act_func = ReLU()
    elif activation_function == 'softmax':
        act_func = Softmax(dim=-1)

    weight = Parameter(torch.empty((out_features, in_features), device=device))
    bias = Parameter(torch.empty(out_features, device=device))

    def weight_and_bias_initialization():
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    weight_and_bias_initialization()

    def linear_computation(x: torch.Tensor):
        return act_func(torch.matmul(x, weight.t()) + bias)
    
    return linear_computation, weight, bias
