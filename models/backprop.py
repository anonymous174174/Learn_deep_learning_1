import torch
import torch.nn as nn
import torch.nn.functional as F

class Backpropagation:
    def cross_entropy_output_grad(self, outputs,targets):
        """
        Compute the partial derivative of the loss with respect to the output pre-activation for cross entropy loss function.
        targets is a (batch_size, num_classes) tensor of one-hot encoded labels.
        outputs is a (batch_size, num_classes) tensor of model outputs with the softmax applied.
        dl/dal (dal is the pre-activation output) = -(targets - outputs)
        This is the gradient of the loss with respect to the pre-activation output.
        """
        gradient = -(targets-outputs)
        return gradient
    def dloss_dactivations_prev(self, weight_matrix,gradient_pre_activation):
        """
        
        Compute the gradient of the loss function with respect to the previous layer's post activation output.
        gradient_pre_activation (batchsize, No of neurons in current layer) is the gradient of the loss with respect to the pre-activation of the current layer.
        dpreactivation current layer is (derivative of pre-activation current layer with respect to the post-activation output of the previous layer)
        dpreactivation current layer/dactivations_prev =  for each of the individual post-activation output of the previous layer is that numbered column of the weight_matrix


        dloss/dactivations_prev = dloss/dpreactivation current layer * dpreactivation current layer/dactivations_prev

        gradient_pre_activation is the gradient of the loss with respect to the pre-activation output of the current layer(dloss/dpreactivation current layer).
        ***Please note that weight matrix shape is (no of neurons in current layer, no of neurons in previous layer) and the gradient_pre_activation shape is (batchsize, no of neurons in current layer)***
        dpreactivation current layer/dactivations_prev for all h = weight_matrix
        dloss/dactivations_prev = gradient_pre_activation @ weight_matrix
        where weight_matrix is the weight matrix between current and previous layers.
        """
        return gradient_pre_activation@weight_matrix
    
    def dloss_dweights(self, activations_prev, gradient_pre_activation):
        """
        Compute the gradient of the loss function with respect to the weights of the current layer.
        activations_prev (batchsize, no of neurons in previous layer) is the post-activation output of the previous layer.
        gradient_pre_activation (batchsize, no of neurons in current layer) is the gradient of the loss with respect to the pre-activation output of the current layer.
        
        dloss/dweights = dloss/dpreactivation current layer * dpreactivation current layer/dweights
        dpreactivation current layer/dweights = activations_prev
        dloss/dweights = gradient_pre_activation.T @ activations_prev
        """
        return activations_prev.T @ gradient_pre_activation
    def gradients_model(self, model, inputs,targets):

        """
        Compute gradients of the model with respect to inputs and targets.
        """

        # Set the model to training mode
        model.train()
        
        # Zero the gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = F.mse_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data)
        
        return gradients
    

    
    pass