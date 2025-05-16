import matplotlib.pyplot as plt

# from models.model_layers import DenseLayer
import torch
from models.neuralnet import DenseNet_classifier
from models.backprop import Backpropagation
from loss.loss_functions import CrossEntropyLoss
from optim.optimizers_weight_decay import SGD, Adam, RMSprop, Nadam, Nesterov, Nadam
import torch
from data.dataloader import MNISTDataLoader
from torch.cuda import device_of
device = "mps" if torch.backends.mps.is_available() else  "cuda" if torch.cuda.is_available() else "cpu"
dataset=MNISTDataLoader(dataset="fashion_mnist",dtype=torch.float32,batch_size=32,device=device,normalize=True)

model=DenseNet_classifier(model_config=[784,128,128,128,128,10],dtype=torch.float32,device=device,weight_init="xavier",activation_hidden_layers="relu",  loss_function="cross_entropy")
opt=Nadam(layers=model.model,lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0000)
lf=CrossEntropyLoss(dtype=torch.float32,device=device)
def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions given one-hot encoded targets.
    
    Parameters:
        predictions (Tensor): (batch_size, num_classes)
        targets (Tensor): (batch_size, num_classes)
    
    Returns:
        float: Accuracy value
    """
    pred_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    correct = (pred_labels == true_labels).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0
# every_epoch_loss=[] 
# loss_store=[]
accuracy=0
for epoch in range(1,11):
    for batch_idx ,(imgs,targets) in enumerate(dataset.get_train_batches(shuffle=True)):
        predictions=model.forward(imgs)
        # loss_batch=lf.forward(predictions=predictions, targets=targets)
        model.calculate_gradients(predictions=predictions, targets=targets)
        opt.step()
        # loss_store.append(loss_batch.item())
        ab=compute_accuracy(predictions, targets)
        #print(f"Accuracy for batch {batch_idx+1}: {ab:.4f}",end='\r')
        accuracy+=ab
        #print(f"Epoch {epoch}/5 | Batch {batch_idx+1}", end='\r')
    print("Training Accuracy for epoch",epoch,"is",accuracy/(batch_idx+1))
    accuracy=0
    for batch_idx1, (img, targets) in enumerate(dataset.get_test_batches()):
        predictions=model.predict(img)
        ab=compute_accuracy(predictions, targets)
        accuracy+=ab
    
    print("Validation Accuracy for epoch",epoch,"is",accuracy/(batch_idx1+1))
    accuracy=0
    print("-------------------------------------------------------------")
    # every_epoch_loss.append(loss_store)
    # loss_batch=[]

# for epoch_idx, epoch_loss in enumerate(every_epoch_loss):
#     plt.figure()
#     plt.plot(epoch_loss, label=f'Epoch {epoch_idx + 1}')
#     plt.xlabel('Batch')
#     plt.ylabel('Loss')
#     plt.title(f'Loss Curve - Epoch {epoch_idx + 1}')
#     plt.legend()
#     plt.savefig(f'epoch_{epoch_idx + 1}_loss.png')  # Save the plot as an image file
#     plt.close()  # Close the figure to free memory

# print("Saved loss plots for each epoch.")
