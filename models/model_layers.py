# layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class DenseLayer:
    def __init__(self, in_features, out_features, dtype,device,weight_init,requires_grad=True):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.weight_init = weight_init
        self.weight = torch.zeros((out_features, in_features), dtype=dtype,device=device)
        self.bias = torch.zeros((out_features,), dtype=dtype,device=device)
        nn.init.xavier_uniform_(self.weight) if weight_init == 'xavier' else nn.init.uniform_(self.weight)
        self.requires_grad = requires_grad

    def forward(self, x):
        if self.requires_grad:
            self.input=x
        x= x@self.weight.T + self.bias
        return x
    def forward_no_grad(self, x):
        x= x@self.weight.T + self.bias
        return x
    def backward(self, grad_output):
        "to implement later"
        pass

    
