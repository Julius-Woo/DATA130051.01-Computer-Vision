<h1 align="center">DATA130051.01 Final Project</h1>
<h3 align="center"> Wu Jia'ao  21307130203 </h3>

## Contents
- [Contents](#contents)
- [Requirements](#requirements)
- [Task 1: Self-supervised and Supervised Learning on Image Classification](#task-1-self-supervised-and-supervised-learning-on-image-classification)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Task 2: CNN vs. Transformer for Image Classification](#task-2-cnn-vs-transformer-for-image-classification)
  - [Data Preparation](#data-preparation-1)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation-1)
  - [Visualization](#visualization-1)
- [Task 3: NeRF: 3D Reconstruction and View Synthesis](#task-3-nerf-3d-reconstruction-and-view-synthesis)
  - [Data Preparation](#data-preparation-2)
  - [Model Training](#model-training-1)
  - [Evaluation](#evaluation-2)
  - [Visualization](#visualization-2)

***

## Requirements

This project requires Python >= 3.8. See the requirements.txt file for the required packages in each task folder. You can install them using the following command:

```cmd
pip install -r requirements.txt
```

> [!NOTE]
> You need to open the task directory (e.g. `SimCLR`) as the working directory to run the scripts and notebooks correctly.

## Task 1: Self-supervised and Supervised Learning on Image Classification

### Data Preparation

Dataset STL-10 and CIFAR-100 are used in this task. You can eitgher:

- Download the datasets from [here](https://cs.stanford.edu/~acoates/stl10/) and [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put them in the `SimCLR/datasets` directory,
- Or run the training command below to download the datasets automatically.

### Training the Model

Run the following command to train the SimCLR model:

```cmd
$ python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 --sub-ratio 0.75
```

The model weights, logs, and TensorBoard events will be saved in the `SimCLR/runs` directory.

### Evaluation

First download the weights for the model from [Final/SimCLR/...](https://pan.baidu.com/s/1mjC8PhPnaIVdDaCg3XPOmQ?pwd=cctv) and put it in the `SimCLR/runs` directory.

Then open the `LCP_evaluator.ipynb` notebook, run all the function cells and the cell `Linear Classification Protocal`.

### Visualization

Run the following command to start the TensorBoard:

```cmd
tensorboard --logdir ./runs/
```

Then open the browser and go to `http://localhost:6006/`.

## Task 2: CNN vs. Transformer for Image Classification

### Data Preparation

Dataset CIFAR-100 is used in this task. You can eitgher:

- Download the datasets from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in the `CNN_Transformer/data` directory,
- Or run the Jupyter Notebook below to download the datasets automatically.

### Model Training

Simply open the `CNN_Transformer_formal.ipynb` notebook and run all the cells.

### Evaluation

First download the weights for the model from [Final/CNN_Transformer/training_results_midmodel/...](https://pan.baidu.com/s/1mjC8PhPnaIVdDaCg3XPOmQ?pwd=cctv) and put it in the `CNN_Transformer/training_results_midmodel` directory.

Then open the `CNN_Transformer_formal.ipynb` notebook, run the cell `Evaluation`.

### Visualization

Run the following command to start the TensorBoard:

```cmd
tensorboard --logdir ./training_results_midmodel
```

Then open the browser and go to `http://localhost:6006/`.

## Task 3: NeRF: 3D Reconstruction and View Synthesis

### Data Preparation

First download [Final/NeRF/data](https://pan.baidu.com/s/1mjC8PhPnaIVdDaCg3XPOmQ?pwd=cctv) to the `NeRF/` directory.

If you want to create your own datasets, please refer to this [tutorial](https://blog.csdn.net/qq_45913887/article/details/132731884). Note that the [LLFF](https://github.com/Fyusion/LLFF.git) repository is required, and is already included in the `NeRF/` directory.

### Model Training

Run the following commands to train NeRF:

```cmd
$ python run_nerf.py --config configs/kipling.txt
$ python run_nerf.py --config configs/steamedbuns.txt
```

### Evaluation

First download [Final/NeRF/{kipling|steamedbuns}](https://pan.baidu.com/s/1mjC8PhPnaIVdDaCg3XPOmQ?pwd=cctv) to the `NeRF/logs` directory.

Then run the following command to render videos with the trained model:

```cmd
$ python run_nerf.py --config configs/kipling.txt --render_only
$ python run_nerf.py --config configs/steamedbuns.txt --render_only
```

### Visualization

Run the following command to start the TensorBoard:

```cmd
tensorboard --logdir ./logs/summaries/
```

Then open the browser and go to `http://localhost:6006/`.