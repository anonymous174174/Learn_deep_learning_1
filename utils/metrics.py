# metrics.py
import torch
import wandb

def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions given one-hot encoded targets.
    
    Parameters:
        predictions (Tensor): (batch_size, num_classes)
        targets (Tensor): (batch_size, num_classes)
    
    Returns:
        float: Accuracy value
    """
    pred_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    correct = (pred_labels == true_labels).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def compute_confusion_matrix(predictions, targets, num_classes):
    """
    Computes the confusion matrix for multi-class classification.
    
    Parameters:
        predictions (Tensor): (batch_size, num_classes)
        targets (Tensor): (batch_size, num_classes)
        num_classes (int): number of classes
        
    Returns:
        Tensor: Confusion matrix of size (num_classes, num_classes)
                where row indices correspond to true classes and columns to predicted classes.
    """
    pred_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int)
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1
    return confusion

def log_metrics(epoch, train_loss, train_accuracy,val_loss,val_accuracy,  additional_metrics=None):

    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
    }

    # if confusion_matrix is not None:
    #     metrics["confusion_matrix"] = confusion_matrix.tolist()  # convert tensor to list for logging
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    wandb.log(metrics)