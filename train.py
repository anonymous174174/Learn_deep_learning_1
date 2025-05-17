import argparse
from models.neuralnet import DenseNet_classifier
from optim.optimizers_weight_decay import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from loss.loss_functions import CrossEntropyLoss, MeanSquaredError
from data.dataloader import MNISTDataLoader
import torch
import time

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with configurable options.")
    
    # Model configuration
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-m', '--momentum', default=0.9, type=float)
    parser.add_argument('-beta', '--beta', default=0.9, type=float)
    parser.add_argument('-beta1', '--beta1', default=0.9, type=float)
    parser.add_argument('-beta2', '--beta2', default=0.999, type=float)
    parser.add_argument('-eps', '--epsilon', default=1e-8, type=float)
    parser.add_argument('-w_d', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('-w_i', '--weight_init', default='xavier', choices=['random', 'xavier'])
    parser.add_argument('-nhl', '--num_layers', default=3, type=int)
    parser.add_argument('-sz', '--hidden_size', default=128, type=int)
    parser.add_argument('-a', '--activation', default='relu', choices=['sigmoid', 'tanh', 'relu'])
    
    return parser.parse_args()

def main():
    args = get_args()
    start_time = time.time()
    
    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32
    print(f"Using device: {device}\n")

    # Load dataset
    dataset = MNISTDataLoader(
        dataset=args.dataset,
        dtype=dtype,
        batch_size=args.batch_size,
        device=device,
        normalize=True
    )

    # Model configuration
    model_config = [args.hidden_size]*args.num_layers
    model_config.insert(0, 784)  # Input size
    model_config.append(10)      # Output size

    # Initialize model
    model = DenseNet_classifier(
        model_config=model_config,
        dtype=dtype,
        device=device,
        weight_init=args.weight_init,
        activation_hidden_layers=args.activation,
        loss_function=args.loss
    )
    print(f"Model architecture:\n input layer: 784 neurons\n hidden dense layers: {model_config[1:-1]}\n output layer: 10 neurons\n")

    # Loss function
    if args.loss == 'cross_entropy':
        loss_fn = CrossEntropyLoss(dtype=dtype, device=device)
    else:
        loss_fn = MeanSquaredError(dtype=dtype, device=device)

    # Optimizer
    optimizer_configs = {
    'sgd': {'class': SGD, 'kwargs': {'weight_decay': args.weight_decay}},
    'momentum': {'class': Momentum, 'kwargs': {'momentum': args.momentum, 'weight_decay': args.weight_decay}},
    'nag': {'class': Nesterov, 'kwargs': {'momentum': args.momentum, 'weight_decay': args.weight_decay}},
    'rmsprop': {'class': RMSprop, 'kwargs': {'beta': args.beta, 'epsilon': args.epsilon, 'weight_decay': args.weight_decay}},
    'adam': {'class': Adam, 'kwargs': {'beta1': args.beta1, 'beta2': args.beta2, 'epsilon': args.epsilon, 'weight_decay': args.weight_decay}},
    'nadam': {'class': Nadam, 'kwargs': {'beta1': args.beta1, 'beta2': args.beta2, 'epsilon': args.epsilon, 'weight_decay': args.weight_decay}}
                        }

    optimizer_class = optimizer_configs[args.optimizer]['class']
    optimizer_kwargs = optimizer_configs[args.optimizer]['kwargs']

    optimizer = optimizer_class(layers=model.model, lr=args.learning_rate, **optimizer_kwargs)

    print(f"Training configuration:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Optimizer: {args.optimizer}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Loss function: {args.loss}\n")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for batch_idx, (imgs, targets) in enumerate(dataset.get_train_batches(shuffle=True)):
            # Forward pass
            predictions = model.forward(imgs)
            loss = loss_fn.forward(predictions, targets)
            
            # Backward pass and optimize
            model.calculate_gradients(predictions, targets)
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            pred_labels = torch.argmax(predictions,dim=1)
            true_labels = torch.argmax(targets,dim=1)
            total += targets.size(0)
            correct += (pred_labels == true_labels).sum().item()

            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{int(dataset.train_size/args.batch_size)} | "
                      f"Loss: {loss.item():.4f}", end='\r')

        # Validation phase
        #model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, targets in dataset.get_test_batches():
                predictions = model.predict(imgs)
                loss = loss_fn.forward(predictions, targets)
                
                val_loss += loss.item()
                pred_labels = torch.argmax(predictions,dim=1)
                true_labels = torch.argmax(targets,dim=1)
                val_total += targets.size(0)
                val_correct += (pred_labels == true_labels).sum().item()

        # Calculate metrics
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / dataset.train_size
        avg_val_loss = val_loss / dataset.test_size

        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{args.epochs} [{epoch_time:.1f}s]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("--------------------------------------------------")

    # Final summary
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()



# import argparse
# from models.neuralnet import DenseNet_classifier
# from optim.optimizers_weight_decay import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
# from loss.loss_functions import CrossEntropyLoss, MeanSquaredError
# from data.dataloader import MNISTDataLoader
# from utils.metrics import compute_accuracy
# import torch
# import wandb
# def get_args():
#     parser = argparse.ArgumentParser(description="Train a neural network with configurable options.")

#     parser.add_argument('-wp', '--wandb_project', default='myprojectname', type=str, help='Project name for Weights & Biases')
#     parser.add_argument('-we', '--wandb_entity', default='myname', type=str, help='Wandb Entity for Weights & Biases')
#     parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
#     parser.add_argument('-e', '--epochs', default=3, type=int, help='Number of epochs')
#     parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
#     parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
#     parser.add_argument('-o', '--optimizer', default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
#     parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate')
#     parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum for optimizers')
#     parser.add_argument('-beta', '--beta', default=0.9, type=float, help='Beta for RMSProp')
#     parser.add_argument('-beta1', '--beta1', default=0.9, type=float, help='Beta1 for Adam/Nadam')
#     parser.add_argument('-beta2', '--beta2', default=0.999, type=float, help='Beta2 for Adam/Nadam')
#     parser.add_argument('-eps', '--epsilon', default=0.000001, type=float, help='Epsilon for optimizers')
#     parser.add_argument('-w_d', '--weight_decay', default=1e-6 , type=float, help='Weight decay')
#     parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'xavier'], help='Weight initialization')
#     parser.add_argument('-nhl', '--num_layers', default=3, type=int, help='Number of hidden layers')
#     parser.add_argument('-sz', '--hidden_size', default=32, type=int, help='Hidden layer size')
#     parser.add_argument('-a', '--activation', default='relu', choices=[ 'sigmoid', 'tanh', 'relu'], help='Activation function') 
#     """Not identity as it doens't introduce non-linearity and is not used in practice"""
#     # parser.add_argument('-d_type', '--data_type', default='float32', choices=['float32', 'float64','float16'], help='Data type for tensors')

#     return parser.parse_args()

# if __name__ == "__main__":
#     args = get_args()
#     wandb.init(
#         project=args.wandb_project,
#         entity=args.wandb_entity,
#         config=vars(args),
#         name=f"{args.optimizer}_bs{args.batch_size}_lr{args.learning_rate}_act_{args.activation}_nhl{args.num_layers}"
#     )

#     # Load data
#     dtype= torch.float32
#     device = "mps" if torch.backends.mps.is_available() else  "cuda" if torch.cuda.is_available() else "cpu"
#     if device=="cpu": print("Warning: Using CPU for training, consider using GPU for faster training.")
#     if args.dataset not in ['mnist', 'fashion_mnist']:
#         raise ValueError("Unsupported dataset. Supported datasets are: mnist, fashion_mnist")
#     dataset=MNISTDataLoader(dataset=args.dataset,dtype=dtype,batch_size=args.batch_size,device=device,normalize=True)
#     model_config=[args.hidden_size]*args.num_layers
#     model_config.insert(0,784) # input size
#     model_config.append(10) # output size
#     model=DenseNet_classifier(model_config=model_config,dtype=dtype,device=device,weight_init=args.weight_init,activation_hidden_layers=args.activation, loss_function=args.loss)
#     if args.loss == 'cross_entropy':
#         loss_function = CrossEntropyLoss(dtype=dtype, device=device)
#     elif args.loss == 'mean_squared_error':
#         loss_function = MeanSquaredError(dtype=dtype, device=device)
#     else:
#         raise ValueError("Unsupported loss function")
#     if args.optimizer == 'sgd':
#         optimizer = SGD(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
#     elif args.optimizer == 'momentum':
#         optimizer = Momentum(model, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#     elif args.optimizer == 'nag':
#         optimizer = Nesterov(model, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#     elif args.optimizer == 'rmsprop':
#         optimizer = RMSprop(model, learning_rate=args.learning_rate, beta=args.beta, epsilon=args.epsilon, weight_decay=args.weight_decay)
#     elif args.optimizer == 'adam':
#         optimizer = Adam(model, learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
#     elif args.optimizer == 'nadam':
#         optimizer = Nadam(model, learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon, weight_decay=args.weight_decay)
#     else:
#         raise ValueError("Unsupported optimizer")
    

#     for epochs in range(1,args.epochs+1):
#         print(f"Epoch {epochs}/{args.epochs}")
#         for batch_idx, (imgs, targets) in enumerate(dataset.get_train_batches(shuffle=True)):
#             predictions=model.forward(imgs)
#             loss_batch=loss_function.forward(predictions=predictions, targets=targets)
#             model.calculate_gradients(predictions=predictions, targets=targets)
#             optimizer.step()
#             if batch_idx % 100 == 0:
#                 print(f"Batch {batch_idx}/{dataset.train_size//args.batch_size} | Loss: {loss_batch.item():.4f}", end='\r')
#         for batch_idx , (imgs, targets) in enumerate(dataset.get_test_batches()):
#             predictions=model.predict(imgs)
#             loss_batch=loss_function.forward(predictions=predictions, targets=targets)
#             if batch_idx % 100 == 0:
#                 print(f"Batch {batch_idx}/{dataset.test_size//args.batch_size} | Loss: {loss_batch.item():.4f}", end='\r')
            

#     print(args)
