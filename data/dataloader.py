from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import torch
import os

class MNISTDataLoader:
    def __init__(self, dataset="fashion_mnist", dtype=torch.float32, batch_size=32, device="cpu", normalize=True):
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.dtype = dtype

        self.cache_file = f"cached_{dataset}_{'norm' if normalize else 'raw'}.pt"

        if os.path.exists(self.cache_file):
            print(f"🔁 Loading cached dataset from {self.cache_file}")
            data = torch.load(self.cache_file)
            self.x_train = data['x_train'].to(device)
            self.y_train = data['y_train'].to(device)
            self.x_test = data['x_test'].to(device)
            self.y_test = data['y_test'].to(device)
            self.labels = data['labels']
        else:
            print(f"💾 Caching dataset to {self.cache_file}")
            self._load_and_cache(dataset)

        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def _load_and_cache(self, dataset):
        if dataset == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            labels = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
        elif dataset == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            labels = [
                "Zero", "One", "Two", "Three", "Four",
                "Five", "Six", "Seven", "Eight", "Nine"
            ]
        else:
            raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

        x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32")
        x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32")

        if self.normalize:
            x_train /= 255.0
            x_test /= 255.0

        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

        self.x_train = torch.tensor(x_train, dtype=self.dtype)
        self.y_train = torch.tensor(y_train, dtype=self.dtype)
        self.x_test = torch.tensor(x_test, dtype=self.dtype)
        self.y_test = torch.tensor(y_test, dtype=self.dtype)

        torch.save({
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test,
            'labels': labels
        }, self.cache_file)

        # Move to device after saving
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.x_test = self.x_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        self.labels = labels

    def get_train_batches(self, shuffle=True):
        indices = torch.randperm(self.train_size) if shuffle else torch.arange(self.train_size)
        for start in range(0, self.train_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def get_batch(self):
        batch_indices = torch.randint(0, self.train_size, (self.batch_size,))
        return self.x_train[batch_indices], self.y_train[batch_indices]

    def get_test_batches(self):
        for start in range(0, self.test_size, self.batch_size):
            end = start + self.batch_size
            yield self.x_test[start:end], self.y_test[start:end]

# class MNISTDataLoader:
#     def __init__(self, dataset="fashion_mnist", dtype=torch.float32, batch_size=32, device="cpu", normalize=True):
#         """
#         dataset: "mnist" or "fashion_mnist"
#         dtype: torch dtype for tensors
#         batch_size: mini-batch size
#         device: "cpu" or "cuda"
#         normalize: whether to scale pixel values to [0, 1]
#         """
#         self.batch_size = batch_size
#         self.device = device
#         self.normalize = normalize

#         if dataset == "fashion_mnist":
#             (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#             self.labels = [
#                             "T-shirt/top",    # 0
#                             "Trouser",        # 1
#                             "Pullover",       # 2
#                             "Dress",          # 3
#                             "Coat",           # 4
#                             "Sandal",         # 5
#                             "Shirt",          # 6
#                             "Sneaker",        # 7
#                             "Bag",            # 8
#                             "Ankle boot"      # 9
#                         ]
#         elif dataset == "mnist":
#             (x_train, y_train), (x_test, y_test) = mnist.load_data()
#             self.labels = [
#                         "Zero",      # 0
#                         "One",       # 1
#                         "Two",       # 2
#                         "Three",     # 3
#                         "Four",      # 4
#                         "Five",      # 5
#                         "Six",       # 6
#                         "Seven",     # 7
#                         "Eight",     # 8
#                         "Nine"       # 9
#                     ]
#         else:
#             raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

#         # Flatten and convert to float
#         x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32")
#         x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32")

#         if self.normalize:
#             x_train /= 255.0
#             x_test /= 255.0

#         # One-hot encode labels
#         y_train = np.eye(10)[y_train]
#         y_test = np.eye(10)[y_test]

#         # Convert to torch tensors
#         self.x_train = torch.tensor(x_train, dtype=dtype, device=device)
#         self.y_train = torch.tensor(y_train, dtype=dtype, device=device)
#         self.x_test = torch.tensor(x_test, dtype=dtype, device=device)
#         self.y_test = torch.tensor(y_test, dtype=dtype, device=device)

#         self.train_size = self.x_train.shape[0]
#         self.test_size = self.x_test.shape[0]

#     def get_train_batches(self, shuffle=True):
#         indices = torch.randperm(self.train_size) if shuffle else torch.arange(self.train_size)
#         for start in range(0, self.train_size, self.batch_size):
#             end = start + self.batch_size
#             batch_indices = indices[start:end]
#             yield self.x_train[batch_indices], self.y_train[batch_indices]
#     def get_batch(self):
#         batch_indices = torch.randint(0, self.train_size, (self.batch_size,))
#         batch = self.x_train[batch_indices], self.y_train[batch_indices]
#         return batch
#     def get_test_batches(self):
#         for start in range(0, self.test_size, self.batch_size):
#             end = start + self.batch_size
#             yield self.x_test[start:end], self.y_test[start:end]
