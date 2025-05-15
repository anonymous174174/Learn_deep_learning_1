# dataloader.py
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import torch

class FashionMNISTDataLoader:
    def __init__(self,dtype, batch_size=32, device="cpu", normalize=True):
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize  


        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32")
        x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32")
        if self.normalize:
            x_train /= 255.0
            x_test /= 255.0

        # one-hot encode labels (Fashion MNIST has 10 classes)
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

        self.x_train = torch.tensor(x_train, dtype=dtype,device=self.device)
        self.y_train = torch.tensor(y_train, dtype=dtype,device=self.device)
        self.x_test = torch.tensor(x_test, dtype=dtype,device=self.device)
        self.y_test = torch.tensor(y_test, dtype=dtype,device=self.device)

        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
    
    def get_train_batches(self, shuffle=True):
        indices = torch.randperm(self.train_size) if shuffle else torch.arange(self.train_size)
        for start in range(0, self.train_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def get_test_batches(self):
        for start in range(0, self.test_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = torch.arange(start, min(end, self.test_size))
            yield self.x_test[batch_indices], self.y_test[batch_indices]