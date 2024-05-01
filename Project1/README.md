<h1 align="center">DATA130051.01 Project 1</h1>
<h3 align="center"> Wu Jia'ao  21307130203 </h3>

## Contents
- [Contents](#contents)
- [File Description](#file-description)
- [Requirements](#requirements)
- [Training the Model](#training-the-model)
- [Hyperparameter Searching](#hyperparameter-searching)
- [Visualization](#visualization)
- [Loading and Testing the Model](#loading-and-testing-the-model)

***

## File Description
- data: Contains Fashion-MNIST dataset. You can also download the dataset from [here](https://github.com/zalandoresearch/fashion-mnist).
- result: Contains the best model parameters and itself.
- fig: Contains the visualization of the training process and some other figures.
- src: Contains the source code.
- requirements.txt: Required packages.



## Requirements
This project requires Python >= 3.8. See the requirements.txt file for the required packages. You can install them using the following command:

```cmd
pip install -r requirements.txt
```

## Training the Model
Download the repository, and set the working directory to the root directory of the project. Run the following command to train the model:

```cmd
python src/main.py
```

The training process will be printed in the console, and the curves of the training process will be presented right after the training process.

To change the hyperparameters, you can modify the `src/main.py` file directly:

```python
n_epoch = 10
batch_size = 128
def create_model(n_neuron_layer=128, learning_rate=1,\
 decay=0.1, moment=0.8, l2_reg_weight=0.0):
    pass
```

To save the model, you can uncomment the last few lines of the `src/main.py` file:

```python
# Save the parameters and the model
model.save_params('./result/fashion_mnist_params.pkl')
save_model(model, './result/fashion_mnist_model.pkl')
print("### Model saved.")
```

The model or only the model's parameters will be saved in the `result` directory.

## Hyperparameter Searching
To search for the best hyperparameters, you can check the examples in the `src/para_tune.ipynb` file. Sequentially run the cells in the notebook and results with different hyperparameters will be printed for comparison.

## Visualization
The accuracy and loss curves are by default plotted immediately after the training process. The following code in `src/main.py` controls the plotting process:

```python
trainer = Trainer(model)
trainer.train(X, y)
trainer.plot_figs()
```

To visualize the weights and biases of the model, you can run the following command, or try the last cell in the `src/para_tune.ipynb` file:

```python
Visualizer.visualize_weights(model.best_weights[0][0])
Visualizer.visualize_biases(model.best_weights[0][1].reshape(-1, 1))
```

## Loading and Testing the Model
To load the model and test it, you can run the following command:

```python
model = load_model('./result/fashion_mnist_model.pkl')
evaluator = Evaluator(model)
evaluator.eval(X_test, y_test, batch_size=128)
```

Alternatively, you can create a new model and load the parameters:

```python
model = Model()
# ...
# Set the model, omitted here
model.load_params('./result/fashion_mnist_params.pkl')
```