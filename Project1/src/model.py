import numpy as np
from layers import InputLayer
from train import CELoss
import pickle


class Model:
    '''Class for a neural network model.'''
    def __init__(self):
        self.layers = []  # List of layers
        self.softmax_output = None
        self.best_weights = None
    
    # Add one layer to the model
    def add(self, layer):
        self.layers.append(layer)
    
    def get_layernum(self):
        return len(self.layers) // 2  # Return number of layers
    
    # Set loss, optimizer and accuracy
    def set_items(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Finalize the model
    def finalize(self):
        self.input_layer = InputLayer()
        layer_count = len(self.layers)
        self.trainlayers = []
        
        # Connect layers correctly
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer  # Dynamic attributes prev and next
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer = self.layers[i]
            
            # Set trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainlayers.append(self.layers[i])
        
        self.loss.set_trainlayers(self.trainlayers)
        self.softmax_output = CELoss()
    
    # Forward pass
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return self.layers[-1].output
    
    # Backpropagation
    def backward(self, output, y):
        self.softmax_output.backward_with_softmax(output, y)
        self.layers[-1].dinputs = self.softmax_output.dinputs
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)
    
    # Retrieve layer parameters, a list of (weights, biases) tuples
    def get_parameters(self):
        parameters = []
        for layer in self.trainlayers:
            parameters.append(layer.get_parameters())
        return parameters
    
    # Set layer parameters
    def set_parameters(self, parameters):
        for layer, parameter in zip(self.trainlayers, parameters):
            layer.set_parameters(*parameter)
    
    # Save parameters to a file
    def save_params(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.get_parameters(), file)
    
    # Load parameters from a file and update the model
    def load_params(self, path):
        with open(path, 'rb') as file:
            self.set_parameters(pickle.load(file))