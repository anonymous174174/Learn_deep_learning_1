import torch
import torch.nn as nn
import torch.nn.functional as F


class Backpropagation:
    def cross_entropy_output_grad(self, predictions,targets):
        """
        Compute the partial derivative of the loss with respect to the output pre-activation for cross entropy loss function.
        targets is a (batch_size, num_classes) tensor of one-hot encoded labels.
        predictions is a (batch_size, num_classes) tensor of model outputs with the softmax applied.
        dl/dal (dal is the pre-activation output) = -(targets - outputs)
        This is the gradient of the loss with respect to the pre-activation output.
        we divide by the batch size because the loss function is averaged over the batch.
        """
        batch_size = predictions.shape[0]
        gradient = -(targets-predictions)/batch_size
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
        dloss/dweights = gradient_pre_activation.T @ activations_prev in a singular case
        but because we are using batch size, we need to sum over the batch size dimension.
        code wise it would basically be
        intermediate_tensor = gradient_pre_activation.unsqueeze(2) * activations_pre.unsqueeze(1)
        shape being (batchsize, no of neurons in current layer, 1) * (batchsize, 1, no of neurons in previous layer)
        => (batchsize, no of neurons in current layer, no of neurons in previous layer)
        final= intermediate.sum(dim=0, keepdim=True)
        another way to do this is to use einsum
        Weights_gradient =torch.einsum('bi,bj->bij', gradient_pre_activation, activations_prev)
        Weights_gradient=Weights_gradient.sum(dim=0,keepdim=True)
        but the most efficient way is to use the @ operator
        gradient_pre_activation.T @ activations_prev
        This works because:
            gradient_pre_activation.T has shape (current_neurons, batch_size),
            activations_prev has shape (batch_size, prev_neurons),
            Their product directly gives the summed gradient over the batch.
        """
        #Weights_gradient = gradient_pre_activation.T @ activations_prev

        return gradient_pre_activation.T @ activations_prev
    
    def dloss_dbias(self, gradient_pre_activation):
        """
        Compute the gradient of the loss function with respect to the bias of the current layer.
        gradient_pre_activation (batchsize, no of neurons in current layer) is the gradient of the loss with respect to the pre-activation output of the current layer.
        dloss/dbias = dloss/dpreactivation current layer * dpreactivation current layer/dbias
        dpreactivation current layer/dbias = 1
        dloss/dbias = gradient_pre_activation
        sum the gradients over the batch size dimension.
        ** Note we don't keep dim because bias is a vector of shape (no of neurons in current layer) and we want the final gradient to be of the same shape.**
        """
        return gradient_pre_activation.sum(dim=0)
    
    def dactivation_dpreactivation(self, activation, activation_function):
        """
        Compute the derivative of the activation function with respect to the pre-activation output for hidden layers till input layer(Not for output layer).
        activation is the post-activation output of any layer.
        activation_function is the activation function used in that layer.
        dactivation/dpreactivation = f'(activation)
        where f' is the derivative of the activation function.
        activation is of shape (batchsize, no of neurons in that layer)
        """
        if activation_function == 'sigmoid':

            return activation * (torch.ones_like(activation) - activation)
        elif activation_function == 'tanh':
            return torch.ones_like(activation) - activation.pow(2)
        elif activation_function == 'relu':
            return (activation > 0).to(activation.dtype)
        else:
            raise ValueError('Unknown activation function')
    
    def gradients_model(self, model, predictions,targets,activation_function_hidden_layers):

        """
        NOTE PASS obj.model of the DenseNet_classifier class as model parameter to this function.
        Compute gradients of the model with respect to inputs and targets 
        and save them in the .grad attribute of the weight and baises tensors.
        inputs is a (batch_size, in_features) tensor of input data.
        targets is a (batch_size, out_features) tensor of target data.

        **Note post_activation_previous_layer = layer.input
            pre_activation_current_layer = layer.input
        """
        
        #gradient_outer_pre_activation_with_respect_to_loss = self.cross_entropy_output_grad(predictions, targets)
        gradient_loss_pre_activation_current_layer =self.cross_entropy_output_grad(predictions, targets) #self.dloss_dactivations_prev(model[-1].weight, gradient_outer_pre_activation_with_respect_to_loss)
        for layer_number in range(model.__len__()-1,-1,-1):
            layer = model[layer_number]
            layer.weight.grad = None
            layer.bias.grad = None
            layer.weight.grad=self.dloss_dweights(layer.input,gradient_loss_pre_activation_current_layer)
            layer.bias.grad=self.dloss_dbias(gradient_loss_pre_activation_current_layer)
            if layer_number > 0:
                gradient_loss_pre_activation_current_layer=self.dloss_dactivations_prev(layer.weight,gradient_loss_pre_activation_current_layer)*self.dactivation_dpreactivation(layer.input,activation_function_hidden_layers)

            

