import numpy as np


class InputLayer:
    '''Class for the input layer in a neural network.'''
    def forward(self, inputs, training):
        self.output = inputs


class FullyConnectedLayer:
    '''Class for one fully connected layer in a neural network.'''
    def __init__(self, n_inputs, n_neurons, l2_reg_weight=0, l2_reg_bias=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l2_reg_weight = l2_reg_weight
        self.l2_reg_bias = l2_reg_bias
    
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # L2 regularization gradients
        if self.l2_reg_weight > 0:
            self.dweights += 2 * self.l2_reg_weight * self.weights
        if self.l2_reg_bias > 0:
            self.dbiases += 2 * self.l2_reg_bias * self.biases
    
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class ActivationReLU:
    '''Class for the ReLU activation function.'''
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs


class ActivationSoftmax:
    '''Class for the softmax activation function.'''
    def forward(self, inputs, training):
        # For numerical stability
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.output = probs
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)  # Create an empty array
        for i, (out, dvalue) in enumerate(zip(self.output, dvalues)):
            out = out.reshape(-1, 1)  # Column vector
            jacobian = np.diagflat(out) - np.dot(out, out.T)
            self.dinputs[i] = np.dot(jacobian, dvalue)
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)