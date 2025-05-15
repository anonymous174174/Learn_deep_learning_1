# loss_functions.py
import torch
class CrossEntropyLoss:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def forward(self, predictions, targets):
        # Ensure predictions are in the range (0, 1)
        """" torch.clamp basically limits the values in the tensor to be within the specified range.
        """

        predictions = torch.clamp(predictions, min=1e-7, max=1 - 1e-7)
        mean_batch_loss = -torch.sum(targets * torch.log(predictions), dim=1).mean()
        return mean_batch_loss
