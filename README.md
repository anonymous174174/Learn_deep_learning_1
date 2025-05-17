# Deep Learning from Scratch

This project implements a simple deep neural network classifier built from scratch using PyTorch. The goal is to demonstrate manual forward and backward propagation, custom loss functions, and various optimization algorithms. The model can be trained on the MNIST or Fashion MNIST datasets.

## Project Structure

- **data/**: Contains the dataloader that loads and preprocesses the MNIST and Fashion MNIST datasets.
- **models/**: Implements the Dense Layer, DenseNet classifier, and backpropagation routines.
- **loss/**: Contains custom loss functions (Cross Entropy and Mean Squared Error).
- **optim/**: Contains various optimizers with weight decay (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam).
- **train.py**: Main training script to run the model.

## Prerequisites

- Python 3.10 or higher
- [PyTorch](https://pytorch.org/)
- [TensorFlow Keras Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets) (for loading MNIST datasets)
- NumPy

## Environment Setup

You have two options to set up the environment:

### Option 1: Using Conda

An `environment.yml` file is provided. To create the environment, run:

```bash
conda env create -f environment.yml
conda activate dl_cpu_env
```

If you need to export your current active Conda environment, you can use:

```bash
conda env export > environment.yml
```

### Option 2: Using pip

A `requirements.txt` file is provided. To set up a virtual environment and install the required packages using pip, run:

```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
pip install -r requirements.txt
```

## How to Run

The training script `train.py` can be run from the command line with various configuration options.

### Basic Training Command

```bash
python train.py
```

By default, the script uses the Fashion MNIST dataset with the following settings:
- **Dataset**: fashion_mnist
- **Epochs**: 10
- **Batch Size**: 64
- **Loss Function**: cross_entropy
- **Optimizer**: adam
- **Learning Rate**: 0.001
- **Weight Initialization**: xavier

### Command Line Options

You can customize the training parameters by passing arguments:

```bash
python train.py --dataset mnist --epochs 20 --batch_size 128 --loss cross_entropy --optimizer adam --learning_rate 0.0005 --weight_init xavier --activation relu --num_layers 3 --hidden_size 128
```

#### Available Options:

- `-d, --dataset`: Select between `"mnist"` and `"fashion_mnist"`.
- `-e, --epochs`: Number of training epochs.
- `-b, --batch_size`: Mini-batch size.
- `-l, --loss`: Loss function (`mean_squared_error` or `cross_entropy`).
- `-o, --optimizer`: Optimizer choice (`sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`, `nadam`).
- `-lr, --learning_rate`: Learning rate.
- `-w, --weight_init`: Weight initialization (`random` or `xavier`).
- Additional options may be available per the command-line argument parser in the `train.py` file.

## Model Overview

- **DenseNet_classifier**: Defines a fully connected network where the architecture is specified by a list (e.g., `[784, 128, 128, 10]`).
- **Activation_functions**: Supports ReLU, Sigmoid, Tanh, and Softmax (for the output layer).
- **Backpropagation**: Implements manual gradient computations for updating weights and biases.
- **Custom Optimizers**: Offer various optimization strategies with support for weight decay.

## Additional Information

- The dataloader preprocesses the data by flattening the images and normalizing the pixel values to the [0, 1] range.
- Labels for **Fashion MNIST** are provided in English (e.g., "T-shirt/top", "Trouser", etc.), and labels for **MNIST** are given as "Zero" to "Nine".

## Running on Different Devices

The code automatically selects the appropriate computation device:
- It uses **CUDA** if available.
- If not, it attempts to use **MPS** (for Apple Silicon).
- Otherwise, it defaults to **CPU**.

## License

This project is provided for educational purposes. Feel free to modify and use it to learn more about deep learning from scratch.

Happy Learning!
