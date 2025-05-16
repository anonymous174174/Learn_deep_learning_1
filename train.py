import argparse
from models.neuralnet import DenseNet_classifier
from optim.optimizers_weight_decay import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from loss.loss_functions import CrossEntropyLoss, MeanSquaredError
from data.dataloader import MNISTDataLoader
from utils.metrics import compute_accuracy
import torch
# import wandb
def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with configurable options.")

    parser.add_argument('-wp', '--wandb_project', default='myprojectname', type=str, help='Project name for Weights & Biases')
    parser.add_argument('-we', '--wandb_entity', default='myname', type=str, help='Wandb Entity for Weights & Biases')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('-e', '--epochs', default=3, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    parser.add_argument('-o', '--optimizer', default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum for optimizers')
    parser.add_argument('-beta', '--beta', default=0.9, type=float, help='Beta for RMSProp')
    parser.add_argument('-beta1', '--beta1', default=0.9, type=float, help='Beta1 for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', default=0.999, type=float, help='Beta2 for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', default=0.000001, type=float, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', default=1e-6 , type=float, help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'xavier'], help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', default=32, type=int, help='Hidden layer size')
    parser.add_argument('-a', '--activation', default='relu', choices=[ 'sigmoid', 'tanh', 'relu'], help='Activation function') 
    """Not identity as it doens't introduce non-linearity and is not used in practice"""
    # parser.add_argument('-d_type', '--data_type', default='float32', choices=['float32', 'float64','float16'], help='Data type for tensors')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # wandb.init(
    #     project=args.wandb_project,
    #     entity=args.wandb_entity,
    #     config=vars(args),
    #     name=f"{args.optimizer}_bs{args.batch_size}_lr{args.learning_rate}_act_{args.activation}_nhl{args.num_layers}"
    # )

    # Load data
    dtype= torch.float32
    device = "mps" if torch.backends.mps.is_available() else  "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cpu": print("Warning: Using CPU for training, consider using GPU for faster training.")
    if args.dataset not in ['mnist', 'fashion_mnist']:
        raise ValueError("Unsupported dataset. Supported datasets are: mnist, fashion_mnist")
    dataset=MNISTDataLoader(dataset=args.dataset,dtype=dtype,batch_size=args.batch_size,device=device,normalize=True)
    model_config=[args.hidden_size]*args.num_layers
    model_config.insert(0,784) # input size
    model_config.append(10) # output size
    model=DenseNet_classifier(model_config=model_config,dtype=dtype,device=device,weight_init=args.weight_init,activation_hidden_layers=args.activation, loss_function=args.loss)
    if args.loss == 'cross_entropy':
        loss_function = CrossEntropyLoss(dtype=dtype, device=device)
    elif args.loss == 'mean_squared_error':
        loss_function = MeanSquaredError(dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported loss function")
    if args.optimizer == 'sgd':
        optimizer = SGD(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'momentum':
        optimizer = Momentum(model, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'nag':
        optimizer = Nesterov(model, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(model, learning_rate=args.learning_rate, beta=args.beta, epsilon=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = Adam(model, learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'nadam':
        optimizer = Nadam(model, learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")
    

    train_loader, val_loader, input_size, num_classes = load_dataset(args.dataset, args.batch_size)

    
    print(args)
    # ...existing code...