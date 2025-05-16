import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with configurable options.")

    parser.add_argument('-wp', '--wandb_project', default='myprojectname', type=str, help='Project name for Weights & Biases')
    parser.add_argument('-we', '--wandb_entity', default='myname', type=str, help='Wandb Entity for Weights & Biases')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('-e', '--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    parser.add_argument('-o', '--optimizer', default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float, help='Learning rate')
    parser.add_argument('-m', '--momentum', default=0.5, type=float, help='Momentum for optimizers')
    parser.add_argument('-beta', '--beta', default=0.5, type=float, help='Beta for RMSProp')
    parser.add_argument('-beta1', '--beta1', default=0.5, type=float, help='Beta1 for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', default=0.5, type=float, help='Beta2 for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', default=0.000001, type=float, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'], help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', default=1, type=int, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', default=4, type=int, help='Hidden layer size')
    parser.add_argument('-a', '--activation', default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    # ...existing code...