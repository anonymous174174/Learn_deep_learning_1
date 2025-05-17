import argparse
from models.neuralnet import DenseNet_classifier
from optim.optimizers_weight_decay import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from loss.loss_functions import CrossEntropyLoss, MeanSquaredError
from data.dataloader import MNISTDataLoader
import torch
import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Train a Dense neural network with configurable options.")
    
    # Wandb configuration
    parser.add_argument('-wp', '--wandb_project', default='my-sweep-project', type=str)
    parser.add_argument('-we', '--wandb_entity', default='my-entity', type=str)
    
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

def train(config=None):
    # Initialize wandb run
    with wandb.init(config=config):
        config = wandb.config
        
        # Setup device and dtype
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32
        
        # Load dataset
        dataset = MNISTDataLoader(
            dataset=config.dataset,
            dtype=dtype,
            batch_size=config.batch_size,
            device=device,
            normalize=True
        )
        
        # Model configuration
        model_config = [config.hidden_size]*config.num_layers
        model_config.insert(0, 784)  # Input size
        model_config.append(10)       # Output size
        
        # Initialize model
        model = DenseNet_classifier(
            model_config=model_config,
            dtype=dtype,
            device=device,
            weight_init=config.weight_init,
            activation_hidden_layers=config.activation,
            loss_function=config.loss
        )
        
        # Loss function
        if config.loss == 'cross_entropy':
            loss_fn = CrossEntropyLoss(dtype=dtype, device=device)
        else:
            loss_fn = MeanSquaredError(dtype=dtype, device=device)
        
        # Optimizer
        optimizer_configs = {
        'sgd': {'class': SGD, 'kwargs': {'weight_decay': config.weight_decay}},
        'momentum': {'class': Momentum, 'kwargs': {'momentum': config.momentum, 'weight_decay': config.weight_decay}},
        'nag': {'class': Nesterov, 'kwargs': {'momentum': config.momentum, 'weight_decay': config.weight_decay}},
        'rmsprop': {'class': RMSprop, 'kwargs': {'beta': config.beta, 'epsilon': config.epsilon, 'weight_decay': config.weight_decay}},
        'adam': {'class': Adam, 'kwargs': {'beta1': config.beta1, 'beta2': config.beta2, 'epsilon': config.epsilon, 'weight_decay': config.weight_decay}},
        'nadam': {'class': Nadam, 'kwargs': {'beta1': config.beta1, 'beta2': config.beta2, 'epsilon': config.epsilon, 'weight_decay': config.weight_decay}}
                            }

        optimizer_class = optimizer_configs[config.optimizer]['class']
        optimizer_kwargs = optimizer_configs[config.optimizer]['kwargs']

        optimizer = optimizer_class(layers=model.model, lr=config.learning_rate, **optimizer_kwargs)
        
        # Training loop
        for epoch in range(1, config.epochs + 1):
            # Training phase
            train_loss = 0.0
            correct = 0
            total = 0
            
            for imgs, targets in dataset.get_train_batches(shuffle=True):
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

            # Validation phase
            # model.eval()
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
            
            # Calculate epoch metrics
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / dataset.train_size
            avg_val_loss = val_loss / dataset.test_size
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })
            
            print(f"Epoch {epoch}/{config.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    args = get_args()
    
    # Wandb sweep configuration
    sweep_config = {
        'method': 'bayes',  # or 'random', 'grid'
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'optimizer': {
                'values': ['adam', 'nadam', 'rmsprop']
            },
            'weight_decay': {
                'min': 0,
                'max': 1e-3
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.wandb_project,
        entity=args.wandb_entity
    )
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train)