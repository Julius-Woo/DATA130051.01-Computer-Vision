import numpy as np
from utils import Visualizer


class SGDOptimizer:
    '''Class for the SGD optimizer.'''

    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.lr = learning_rate
        self.cur_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Decay learning rate before parameter update
    def pre_update(self):
        if self.decay:
            self.cur_lr = self.lr * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:  # Use momentum
            if not hasattr(layer, 'weight_momentums'):
                # Initialize cache for momentum
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * \
                layer.weight_momentums - self.cur_lr * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.cur_lr * layer.dbiases
            layer.bias_momentums = bias_updates
        else:  # Vanilla SGD
            weight_updates = -self.cur_lr * layer.dweights
            bias_updates = -self.cur_lr * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self):
        self.iterations += 1


class Loss:
    '''Class for common loss.'''

    # Set trainable layers, i.e. layers with weights
    def set_trainlayers(self, layers):
        self.trainlayers = layers

    # Regularization loss
    def reg_loss(self):
        reg_loss = 0
        for layer in self.trainlayers:
            if layer.l2_reg_weight > 0:
                reg_loss += layer.l2_reg_weight * \
                    np.sum(layer.weights * layer.weights)
            if layer.l2_reg_bias > 0:
                reg_loss += layer.l2_reg_bias * \
                    np.sum(layer.biases * layer.biases)
        return reg_loss

    # Calculate data loss and regularization loss
    def calculate(self, output, y, include_reg=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.cum_loss += np.sum(sample_losses)
        self.cum_count += len(sample_losses)
        if not include_reg:
            return data_loss
        return data_loss, self.reg_loss()

    # Calculate accumulated loss
    def calc_cum_loss(self, include_reg=False):
        data_loss = self.cum_loss / self.cum_count
        if not include_reg:
            return data_loss
        return data_loss, self.reg_loss()

    def reset_cum_loss(self):
        self.cum_loss = 0
        self.cum_count = 0


class CELoss(Loss):
    '''Class for the cross-entropy loss.'''

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid division by zero
        if len(y_true.shape) == 1:  # Target is a single value for each sample
            # Confidence values for target values
            confidences = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:  # One-hot encoded, a row for each sample
            confidences = np.sum(y_pred * y_true, axis=1)
        return -np.log(confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:  # Turn to one-hot encoded
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples  # Normalize gradient

    # Backward pass with softmax activation for the last layer
    def backward_with_softmax(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Trainer():
    def __init__(self, model):
        self.model = model
        self.loss_train = []
        self.acc_train = []
        self.lr_history = []
        self.loss_val = []
        self.acc_val = []
        self.best_acc = 0.0

    def train(self, X, y, epochs=1, batch_size=None, print_every=1, val_data=None):
        # Initialization
        train_steps = 1  # Default step if batch size is not set
        if val_data is not None:
            val_steps = 1
            X_val, y_val = val_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if val_data is not None:
                val_steps = len(X_val) // batch_size
                if val_steps * batch_size < len(X_val):
                    val_steps += 1

        # Main loop
        for epoch in range(epochs):
            if print_every:
                print(f'Epoch {epoch + 1}/{epochs}')
            self.model.loss.reset_cum_loss()
            self.model.accuracy.reset_cum()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Forward pass
                output = self.model.forward(batch_X, training=True)
                data_loss, reg_loss = self.model.loss.calculate(
                    output, batch_y, include_reg=True)
                loss = data_loss + reg_loss
                predictions = self.model.output_layer.predictions(output)
                accuracy = self.model.accuracy.calculate(predictions, batch_y)

                # Backward pass
                self.model.backward(output, batch_y)

                # Update parameters
                self.model.optimizer.pre_update()
                for layer in self.model.trainlayers:
                    self.model.optimizer.update_params(layer)
                self.model.optimizer.post_update()

                self.loss_train.append(loss)
                self.acc_train.append(accuracy)
                self.lr_history.append(self.model.optimizer.cur_lr)
                # Print loss and accuracy
                if print_every:
                    if not step % print_every or step == train_steps - 1:
                        print(f'step: {step}, ' +
                            f'acc: {accuracy:.4f}, ' +
                            f'loss: {loss:.4f} (' +
                            f'data_loss: {data_loss:.4f}, ' +
                            f'reg_loss: {reg_loss:.4f}), ' +
                            f'lr: {self.model.optimizer.cur_lr:.5f}')
            # Print epoch loss and accuracy
            epoch_data_loss, epoch_reg_loss = self.model.loss.calc_cum_loss(
                include_reg=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.model.accuracy.calc_cum()

            if print_every:
                print(f'Training--Epoch {epoch + 1}/{epochs}, ' +
                    f'acc: {epoch_accuracy:.4f}, ' +
                    f'loss: {loss:.4f} (' +
                    f'data_loss: {data_loss:.4f}, ' +
                    f'reg_loss: {reg_loss:.4f}), ' +
                    f'lr: {self.model.optimizer.cur_lr:.5f}')
            elif epoch == epochs - 1:
                print(f'Training--' +
                    f'acc: {epoch_accuracy:.4f}, ' +
                    f'loss: {loss:.4f} (' +
                    f'data_loss: {data_loss:.4f}, ' +
                    f'reg_loss: {reg_loss:.4f})')
            # Validation set evaluation
            if val_data is not None:
                self.model.loss.reset_cum_loss()
                self.model.accuracy.reset_cum()

                for step in range(val_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]
                    # Forward pass
                    output = self.model.forward(batch_X, training=False)
                    self.model.loss.calculate(output, batch_y)
                    predictions = self.model.output_layer.predictions(output)
                    self.model.accuracy.calculate(predictions, batch_y)
                val_loss = self.model.loss.calc_cum_loss()
                val_accuracy = self.model.accuracy.calc_cum()
                self.loss_val.append(val_loss)
                self.acc_val.append(val_accuracy)
                if val_accuracy > self.best_acc:
                    self.best_acc = val_accuracy
                    self.model.best_weights = self.model.get_parameters()
                if print_every:
                    print(f'Validation--' +
                        f'acc: {val_accuracy:.4f}, ' +
                        f'loss: {val_loss:.4f}')
                elif epoch == epochs - 1:
                    print(f'Validation--' +
                        f'acc: {val_accuracy:.4f}, ' +
                        f'loss: {val_loss:.4f}')
        print('### Training finished!')

    def plot_figs(self):
        Visualizer.plot_loss(self.loss_train, self.loss_val)
        Visualizer.plot_accuracy(self.acc_train, self.acc_val)
        Visualizer.plot_learning_rate(self.lr_history)
