import torch
import unittest
import sys
import os
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backprop import Backpropagation

class TestBackpropagation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bp = Backpropagation()
        # Create random tensors with batch size of 5 and 3 neurons for current layer
        self.batch_size = 5
        self.curr_neurons = 3
        self.prev_neurons = 4
        
        # For cross entropy output grad test
        self.predictions = torch.softmax(torch.randn(self.batch_size, self.curr_neurons), dim=1)
        # Create one-hot encoded targets (random indices)
        indices = torch.randint(0, self.curr_neurons, (self.batch_size,))
        self.targets = torch.zeros_like(self.predictions)
        self.targets[torch.arange(self.batch_size), indices] = 1
        
        # For other tests
        self.weight_matrix = torch.randn(self.curr_neurons, self.prev_neurons)
        self.gradient_pre_activation = torch.randn(self.batch_size, self.curr_neurons)
        self.activations_prev = torch.randn(self.batch_size, self.prev_neurons)
        # For activation derivative tests, we use a random activation output
        self.activation = torch.sigmoid(torch.randn(self.batch_size, self.curr_neurons))

    def test_cross_entropy_output_grad(self):
        # Expected gradient: -(targets - predictions)
        expected = -(self.targets - self.predictions)/self.batch_size
        result = self.bp.cross_entropy_output_grad(self.predictions, self.targets)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "cross_entropy_output_grad did not match expected result.")

    def test_dloss_dactivations_prev(self):
        # Expected: gradient_pre_activation @ weight_matrix
        expected = self.gradient_pre_activation @ self.weight_matrix
        result = self.bp.dloss_dactivations_prev(self.weight_matrix, self.gradient_pre_activation)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "dloss_dactivations_prev did not match expected result.")

    def test_dloss_dweights(self):
        # Expected: gradient_pre_activation.T @ activations_prev
        expected = self.gradient_pre_activation.T @ self.activations_prev
        result = self.bp.dloss_dweights(self.activations_prev, self.gradient_pre_activation)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "dloss_dweights did not match expected result.")

    def test_dloss_dbias(self):
        # Expected: sum along dim=0 of gradient_pre_activation
        expected = self.gradient_pre_activation.sum(dim=0)
        result = self.bp.dloss_dbias(self.gradient_pre_activation)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "dloss_dbias did not match expected result.")

    def test_dactivation_dpreactivation_sigmoid(self):
        # For sigmoid, derivative = activation * (1 - activation)
        expected = self.activation * (1 - self.activation)
        result = self.bp.dactivation_dpreactivation(self.activation, 'sigmoid')
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "dactivation_dpreactivation (sigmoid) did not match expected result.")
    
    def test_dactivation_dpreactivation_tanh(self):
        # For tanh, derivative = 1 - activation^2. Use tanh activation.
        act = torch.tanh(torch.randn(self.batch_size, self.curr_neurons))
        expected = 1 - act.pow(2)
        result = self.bp.dactivation_dpreactivation(act, 'tanh')
        self.assertTrue(torch.allclose(result, expected, atol=1e-6),
                        "dactivation_dpreactivation (tanh) did not match expected result.")
    
    def test_dactivation_dpreactivation_relu(self):
        # For relu, derivative = 1 for positive activation and 0 for non-positive.
        act = torch.randn(self.batch_size, self.curr_neurons)
        expected = (act > 0).to(act.dtype)
        result = self.bp.dactivation_dpreactivation(act, 'relu')
        self.assertTrue(torch.equal(result, expected),
                        "dactivation_dpreactivation (relu) did not match expected result.")
    
    def test_gradients_model_single_layer(self):
        # Test the gradients_model function for a single layer model.
        # Create a dummy "model" list with one layer.
        class DummyLayer:
            def __init__(self, weight, bias, input):
                self.weight = weight
                self.bias = bias
                self.input = input
        
        # Create dummy parameters that require grad.
        dummy_weight = torch.randn(self.curr_neurons, self.prev_neurons, requires_grad=True)
        dummy_bias = torch.randn(self.curr_neurons, requires_grad=True)
        # Create a dummy input and simulate that it was saved at forward pass.
        dummy_input = torch.randn(self.batch_size, self.prev_neurons)
        layer = DummyLayer(weight=dummy_weight, bias=dummy_bias, input=dummy_input)
        
        # Forward pass: compute linear output = input @ weight.T + bias
        predictions = dummy_input @ dummy_weight.t() + dummy_bias 
        # Apply softmax to simulate output
        predictions = torch.softmax(predictions, dim=1)
        # Create one-hot targets
        indices = torch.randint(0, self.curr_neurons, (self.batch_size,))
        targets = torch.zeros_like(predictions)
        targets[torch.arange(self.batch_size), indices] = 1
        
        # Compute gradient at output using our cross entropy grad method.
        grad_output = self.bp.cross_entropy_output_grad(predictions, targets)
        
        # Simulate gradients_model on a single layer.
        # Reset grad fields
        layer.weight.grad = None
        layer.bias.grad = None
        layer.weight.grad = self.bp.dloss_dweights(dummy_input, grad_output)
        layer.bias.grad = self.bp.dloss_dbias(grad_output)
        
        # Now, compute expected gradients using autograd.
        dummy_weight2 = dummy_weight.clone().detach().requires_grad_(True)
        dummy_bias2 = dummy_bias.clone().detach().requires_grad_(True)
        # Forward pass with autograd:
        predictions2 = dummy_input @ dummy_weight2.t() + dummy_bias2
        predictions2 = torch.softmax(predictions2, dim=1)
        # Compute cross entropy: because targets are one-hot, use negative log likelihood manually.
        log_preds = torch.log(predictions2 + 1e-9)
        loss = -(targets * log_preds).sum()/self.batch_size
        loss.backward()
        
        # Compare gradients for weight and bias.
        self.assertTrue(torch.allclose(layer.weight.grad, dummy_weight2.grad, atol=1e-6),
                        "gradients_model: computed weight gradient did not match autograd result.")
        self.assertTrue(torch.allclose(layer.bias.grad, dummy_bias2.grad, atol=1e-6),
                        "gradients_model: computed bias gradient did not match autograd result.")
        
    def test_gradients_model_multi_layer(self):
        # Create a 2-layer model (input_size=4, hidden_size=5, output_size=3)
        input_size = 4
        hidden_size = 5
        output_size = 3
        batch_size = 5
        class DummyLayer:
            def __init__(self, weight, bias):
                self.weight = weight
                self.bias = bias
                self.input = None  # Previous layer's post-activation output
                self.pre_activation = None  # Current layer's pre-activation

            def forward(self, x):
                self.input = x.detach()  # Store input (a_{l-1})
                self.pre_activation = x @ self.weight.T + self.bias  # Store z_l
                return self.pre_activation
        
        # Initialize custom model
        W1 = torch.randn(hidden_size, input_size, requires_grad=True)
        b1 = torch.randn(hidden_size, requires_grad=True)
        W2 = torch.randn(output_size, hidden_size, requires_grad=True)
        b2 = torch.randn(output_size, requires_grad=True)
        
        layer1 = DummyLayer(W1, b1)
        layer2 = DummyLayer(W2, b2)
        custom_model = [layer1, layer2]
        
        # Forward pass with custom model
        x = torch.randn(batch_size, input_size)
        z1 = layer1.forward(x)
        a1 = torch.relu(z1)
        z2 = layer2.forward(a1)
        predictions = torch.softmax(z2, dim=1)
        
        # Create targets
        targets = torch.zeros_like(predictions)
        targets[torch.arange(batch_size), torch.randint(0, output_size, (batch_size,))] = 1
        
        # Compute custom gradients
        bp = Backpropagation()
        bp.gradients_model(custom_model, predictions, targets, 'relu')
        
        # Compute PyTorch gradients
        torch_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        with torch.no_grad():
            torch_model[0].weight.copy_(W1)
            torch_model[0].bias.copy_(b1)
            torch_model[2].weight.copy_(W2)
            torch_model[2].bias.copy_(b2)
        
        torch_pred = torch_model(x)
        loss = nn.CrossEntropyLoss()(torch_pred, targets.argmax(dim=1))
        loss.backward()
        
        # Compare gradients
        self.assertTrue(torch.allclose(custom_model[0].weight.grad, torch_model[0].weight.grad, atol=1e-6),
                        "Input layer weight gradients mismatch")
        self.assertTrue(torch.allclose(custom_model[0].bias.grad, torch_model[0].bias.grad, atol=1e-6),
                        "Input layer bias gradients mismatch")
        self.assertTrue(torch.allclose(custom_model[1].weight.grad, torch_model[2].weight.grad, atol=1e-6),
                        "Output layer weight gradients mismatch")
        self.assertTrue(torch.allclose(custom_model[1].bias.grad, torch_model[2].bias.grad, atol=1e-6),
                        "Output layer bias gradients mismatch")
        print("Custom gradients:")
        print("Input layer weight grad:", custom_model[0].weight.grad)
        print("Input layer bias grad:", custom_model[0].bias.grad)
        print("Output layer weight grad:", custom_model[1].weight.grad)
        print("Output layer bias grad:", custom_model[1].bias.grad)
        print("PyTorch gradients:")
        print("Input layer weight grad:", torch_model[0].weight.grad)
        print("Input layer bias grad:", torch_model[0].bias.grad)
        print("Output layer weight grad:", torch_model[2].weight.grad)
        print("Output layer bias grad:", torch_model[2].bias.grad)

if __name__ == '__main__':
    unittest.main()