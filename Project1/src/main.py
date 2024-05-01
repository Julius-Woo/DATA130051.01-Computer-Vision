import numpy as np
from utils import DataHandler, save_model, Visualizer
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSoftmax
from train import SGDOptimizer, CELoss, Trainer
from evaluate import Evaluator
from metrics import Accuracy


np.random.seed(0)
# fashion_mnist_labels = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }

# Create dataset
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_mnist()
X, y, X_val, y_val = data_handler.split_validation(
    X_train, y_train, val_ratio=0.25)
X, X_test = data_handler.scale(X, X_test)
X_val = data_handler.scale(X_val)

# Use the optimal hyperparameters after search
n_epoch = 10
batch_size = 128
def create_model(n_neuron_layer=128, learning_rate=1,
                 decay=0.1, moment=0.8, l2_reg_weight=0.0):
    # Instantiate the model
    model = Model()
    # 3-layer neural network
    model.add(FullyConnectedLayer(X.shape[1], n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    model.add(FullyConnectedLayer(
        n_neuron_layer, n_neuron_layer, l2_reg_weight))
    model.add(ActivationReLU())
    model.add(FullyConnectedLayer(n_neuron_layer, 10, l2_reg_weight))
    model.add(ActivationSoftmax())
    print("### Number of layers:", model.get_layernum())
    
    # Set loss, optimizer and accuracy objects
    model.set_items(loss=CELoss(), optimizer=SGDOptimizer(
        learning_rate, decay, moment), accuracy=Accuracy())
    model.finalize()

    return model

model = create_model()

# Train the model
trainer = Trainer(model)
trainer.train(X, y, epochs=n_epoch, batch_size=batch_size,
              print_every=100, val_data=(X_val, y_val))
model.set_parameters(model.best_weights)
# Test the model
evaluator = Evaluator(model)
evaluator.eval(X_test, y_test, batch_size=128)

# Plot loss, accuracy and learning rate
trainer.plot_figs()

# Plot best-weight
# Visualizer.visualize_weights(model.best_weights[0][0])

# # Save the parameters and the model
# model.save_params('./result/fashion_mnist_params.pkl')
# save_model(model, './result/fashion_mnist_model.pkl')
# print("### Model saved.")
