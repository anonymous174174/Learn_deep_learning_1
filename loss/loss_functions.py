# loss_functions.py
import torch
class CrossEntropyLoss:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def forward(self, predictions, targets):
        mean_batch_loss = -torch.sum(targets * torch.log(predictions), dim=1).mean()
        return mean_batch_loss

class MeanSquaredError:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def forward(self, predictions, targets):
        mean_batch_loss = torch.mean(torch.square(predictions - targets))
        return mean_batch_loss