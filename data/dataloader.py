from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import torch

class MNISTDataLoader:
    def __init__(self, dataset="fashion_mnist", dtype=torch.float32, batch_size=32, device="cpu", normalize=True):
        """
        dataset: "mnist" or "fashion_mnist"
        dtype: torch dtype for tensors
        batch_size: mini-batch size
        device: "cpu" or "cuda"
        normalize: whether to scale pixel values to [0, 1]
        """
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize

        if dataset == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        elif dataset == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        else:
            raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

        # Flatten and convert to float
        x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32")
        x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32")

        if self.normalize:
            x_train /= 255.0
            x_test /= 255.0

        # One-hot encode labels
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

        # Convert to torch tensors
        self.x_train = torch.tensor(x_train, dtype=dtype, device=device)
        self.y_train = torch.tensor(y_train, dtype=dtype, device=device)
        self.x_test = torch.tensor(x_test, dtype=dtype, device=device)
        self.y_test = torch.tensor(y_test, dtype=dtype, device=device)

        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def get_train_batches(self, shuffle=True):
        indices = torch.randperm(self.train_size) if shuffle else torch.arange(self.train_size)
        for start in range(0, self.train_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            yield self.x_train[batch_indices], self.y_train[batch_indices]
    def get_batch(self):
        batch_indices = torch.randint(0, self.train_size, (self.batch_size,))
        batch = self.x_train[batch_indices], self.y_train[batch_indices]
        return batch
    def get_test_batches(self):
        for start in range(0, self.test_size, self.batch_size):
            end = start + self.batch_size
            yield self.x_test[start:end], self.y_test[start:end]
