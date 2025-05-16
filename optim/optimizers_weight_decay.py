from abc import ABC, abstractmethod
import torch

class Optimizer(ABC):
    """Abstract base class for all optimizers"""
    def __init__(self, layers, lr, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, layers, lr, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)

    def step(self):
        with torch.no_grad():
            for layer in self.layers:
                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)
                layer.weight.data -= self.lr * layer.weight.grad
                layer.bias.data -= self.lr * layer.bias.grad

class Momentum(Optimizer):
    def __init__(self, layers, lr, beta=0.9, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        self.velocities = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]

    def step(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                self.velocities[i]['weights'] = (
                    self.beta * self.velocities[i]['weights'] +
                    (1 - self.beta) * layer.weight.grad
                )
                self.velocities[i]['bias'] = (
                    self.beta * self.velocities[i]['bias'] +
                    (1 - self.beta) * layer.bias.grad
                )
                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)
                layer.weight.data -= self.lr * self.velocities[i]['weights']
                layer.bias.data -= self.lr * self.velocities[i]['bias']

class Nesterov(Optimizer):
    def __init__(self, layers, lr, beta=0.9, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        self.velocities = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]

    def step(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                self.velocities[i]['weights'] = (
                    self.beta * self.velocities[i]['weights'] +
                    (1 - self.beta) * layer.weight.grad
                )
                self.velocities[i]['bias'] = (
                    self.beta * self.velocities[i]['bias'] +
                    (1 - self.beta) * layer.bias.grad
                )
                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)
                layer.weight.data -= self.lr * (
                    self.beta * self.velocities[i]['weights'] +
                    (1 - self.beta) * layer.weight.grad
                )
                layer.bias.data -= self.lr * (
                    self.beta * self.velocities[i]['bias'] +
                    (1 - self.beta) * layer.bias.grad
                )
class RMSprop(Optimizer):
    def __init__(self, layers, lr, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.s = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]

    def step(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                self.s[i]['weights'] = (
                    self.beta * self.s[i]['weights'] +
                    (1 - self.beta) * torch.square(layer.weight.grad)
                )
                self.s[i]['bias'] = (
                    self.beta * self.s[i]['bias'] +
                    (1 - self.beta) * torch.square(layer.bias.grad)
                )
                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)
                layer.weight.data -= self.lr * layer.weight.grad / (torch.sqrt(self.s[i]['weights']) + self.epsilon)
                layer.bias.data -= self.lr * layer.bias.grad / (torch.sqrt(self.s[i]['bias']) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, layers, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]
        self.v = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                # Update moments
                self.m[i]['weights'] = (
                    self.beta1 * self.m[i]['weights'] +
                    (1 - self.beta1) * layer.weight.grad
                )
                self.m[i]['bias'] = (
                    self.beta1 * self.m[i]['bias'] +
                    (1 - self.beta1) * layer.bias.grad
                )
                self.v[i]['weights'] = (
                    self.beta2 * self.v[i]['weights'] +
                    (1 - self.beta2) * torch.square(layer.weight.grad)
                )
                self.v[i]['bias'] = (
                    self.beta2 * self.v[i]['bias'] +
                    (1 - self.beta2) * torch.square(layer.bias.grad)
                )

                # Bias correction
                m_hat_w = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[i]['bias'] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[i]['bias'] / (1 - self.beta2 ** self.t)

                # Decoupled weight decay
                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)

                # Update
                layer.weight.data -= self.lr * m_hat_w / (torch.sqrt(v_hat_w) + self.epsilon)
                layer.bias.data -= self.lr * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, layers, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(layers, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]
        self.v = [{
            'weights': torch.zeros_like(layer.weight.data),
            'bias': torch.zeros_like(layer.bias.data)
        } for layer in layers]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                grad_w = layer.weight.grad
                grad_b = layer.bias.grad

                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * grad_w
                self.m[i]['bias'] = self.beta1 * self.m[i]['bias'] + (1 - self.beta1) * grad_b

                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * grad_w**2
                self.v[i]['bias'] = self.beta2 * self.v[i]['bias'] + (1 - self.beta2) * grad_b**2

                m_hat_w = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[i]['bias'] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[i]['bias'] / (1 - self.beta2 ** self.t)

                # Dozat-style m_bar (Nesterov acceleration)
                m_bar_w = self.beta1 * m_hat_w + (1 - self.beta1) * grad_w
                m_bar_b = self.beta1 * m_hat_b + (1 - self.beta1) * grad_b

                if self.weight_decay > 0:
                    layer.weight.data *= (1 - self.lr * self.weight_decay)

                layer.weight.data -= self.lr * m_bar_w / (torch.sqrt(v_hat_w) + self.epsilon)
                layer.bias.data -= self.lr * m_bar_b / (torch.sqrt(v_hat_b) + self.epsilon)
