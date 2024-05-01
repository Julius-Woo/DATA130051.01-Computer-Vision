# This file contains the utility functions for the project.
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


def mnist_reader(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


class DataHandler:
    '''Class for loading and preprocessing data.'''
    
    def __init__(self, path=None):
        self.path = path
    
    # Loads a MNIST dataset
    def load_mnist(self, path=None):
        if path is None:
            path = self.path
        X_train, y_train = mnist_reader(path, kind='train')
        X_test, y_test = mnist_reader(path, kind='t10k')

        return X_train, y_train, X_test, y_test
    
    # Shuffle the training dataset
    def shuffle(self, X, y):
        keys = np.arange(X.shape[0])
        np.random.shuffle(keys)
        return X[keys], y[keys]

    # Scale samples to [-1, 1]
    def scale(self, X, X_test=None):
        if X_test is None:
            return (X - 127.5) / 127.5
        else:
            X = (X - 127.5) / 127.5
            X_test = (X_test - 127.5) / 127.5
            return X, X_test
    
    # Split a portion of the training set for validation
    def split_validation(self, X_train, y_train, val_ratio=0.25):
        total_size = X_train.shape[0]
        validation_size = int(total_size * val_ratio)
        # Shuffle the dataset before splitting
        X_train, y_train = self.shuffle(X_train, y_train)
        X_val = X_train[:validation_size]
        y_val = y_train[:validation_size]
        X_train_new = X_train[validation_size:]
        y_train_new = y_train[validation_size:]
        return X_train_new, y_train_new, X_val, y_val


def save_model(model, path):
    '''Save a model to a file.'''
    model = copy.deepcopy(model)
    # Clear properties
    model.loss.reset_cum_loss()
    model.accuracy.reset_cum()
    model.input_layer.__dict__.pop('output', None)
    model.loss.__dict__.pop('dinputs', None)
    for layer in model.layers:
        for property in ['inputs', 'output', 'dinputs',
                         'dweights', 'dbiases']:
            layer.__dict__.pop(property, None)
    
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    '''Load a model from a file.'''
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


class Visualizer:
    '''Class for visualization.'''
    @staticmethod
    def plot_loss(train_loss, val_loss=None):
        train_steps = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label='Train Loss')
        if val_loss:
            epoch_length = len(train_loss) // len(val_loss)
            val_steps = range(epoch_length, len(train_loss) + 1, epoch_length)
            plt.plot(val_steps, val_loss, label='Validation Loss',
                     linestyle='-', marker='o')
        plt.title('Loss over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(train_accuracy, val_accuracy=None):
        train_steps = range(1, len(train_accuracy) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_accuracy, label='Train Accuracy')
        if val_accuracy:
            epoch_length = len(train_accuracy) // len(val_accuracy)
            val_steps = range(epoch_length, len(
                train_accuracy) + 1, epoch_length)
            plt.plot(val_steps, val_accuracy,
                     label='Validation Accuracy', linestyle='-', marker='o')
        plt.title('Accuracy over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_learning_rate(lr_history):
        steps = range(1, len(lr_history) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, lr_history, label='Learning Rate')
        plt.title('Learning Rate over steps')
        plt.xlabel('step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_weights(weights):
        num_neurons = weights.shape[1]

        input_dim = weights.shape[0]
        side_length = int(np.sqrt(input_dim))

        fig, axes = plt.subplots(8, 16, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < num_neurons:
                image = weights[:, i].reshape(side_length, side_length)
                ax.imshow(image, cmap='viridis', aspect='auto',
                          interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

        plt.suptitle('Visualization of the First Layer Weights')
        plt.show()
    
    @staticmethod
    def visualize_biases(biases):
        plt.figure(figsize=(10, 6))
        plt.hist(biases, bins=20, color='steelblue', edgecolor='black', alpha=0.75)
        plt.title('Histogram of Biases in the First Layer')
        plt.xlabel('Bias Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
