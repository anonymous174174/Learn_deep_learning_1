# neuralnet.py
from layers import DenseLayer
import torch.nn.functional as F
import torch
from backprop import Backpropagation
class DenseNet_classifier:
    """model_config: list of numbers [input_size,hiddensize1,hiddensize2,...,output_size]"""
    def __init__(self, model_config,dtype,device,weight_init,activation_hidden_layers='relu',  loss_function='cross_entropy'):
        self.model_config = model_config
        self.dtype = dtype
        self.device = device
        self.weight_init = weight_init
        self.activation_hidden_layers= activation_hidden_layers
        # self.optimizer = optimizer
        self.loss_function = loss_function
        self.activation=Activation_functions()
        self.model=[]
        for i in range(model_config.__len__()-1):
            self.model.append(DenseLayer(in_features=model_config[i], out_features=model_config[i+1], dtype=self.dtype,device=self.device,weight_init=self.weight_init))



    def forward(self, x):
        for layer_number in range(len(self.model)-1):
            x = self.model[layer_number].forward(x)
            x = self.activation.apply(x,self.activation_hidden_layers)
        x = self.model[-1].forward(x)
        x = self.activation.apply(x,'softmax')
        return x

    def calculate_gradients(self,predictions,targets): 
        Backpropagation().gradients_model(model=self.model, predictions=predictions, targets=targets, activation_function_hidden_layers=self.activation_hidden_layers,loss_function=self.loss_function)

    def predict(self, x):
        for layer_number in range(len(self.model)-1):
            x= self.model[layer_number].forward_no_grad(x)
            x = self.activation.apply(x,self.activation_hidden_layers)
        x = self.model[-1].forward_no_grad(x)
        x = self.activation.apply(x,'softmax')
        return x


class Activation_functions:
    def apply(self, x,activation_function):
        if activation_function == 'relu':
            return F.relu(x)
        elif activation_function == 'sigmoid':
            return torch.sigmoid(x)
        elif activation_function == 'tanh':
            return torch.tanh(x)
        elif activation_function == 'softmax':
            return F.softmax(x, dim=1)
        # elif activation_function == 'identity':
        #     return x
        else:
            raise ValueError("Unsupported activation function")