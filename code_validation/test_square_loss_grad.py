if __name__ == "__main__":

    import torch

    # Inputs
    batch_size, num_classes = 3, 4
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    t = torch.rand(batch_size, num_classes)
    t = t / t.sum(dim=1, keepdim=True)  # Normalize targets to sum to 1

    # Forward pass
    m = torch.softmax(logits, dim=1)
    loss = torch.sum((m - t) ** 2)/batch_size  # MSE loss

    # Backward pass (autograd)
    loss.backward()
    autograd_grad = logits.grad.clone()

    # Manual gradient
    delta = m - t
    dot = (m * delta).sum(dim=1, keepdim=True)
    gradient = 2 * m * (delta - dot)
    batch_size = m.shape[0]
    gradient /= batch_size
    manual_grad = gradient#2 * m * ((m - t) - (m * (m - t)).sum(dim=1, keepdim=True))

    # Verification
    print("Autograd and manual gradients match:", torch.allclose(autograd_grad, manual_grad))
