<h1 align="center">DATA130051.01 Final Project</h1>
<h3 align="center"> Wu Jia'ao  21307130203 </h3>

- [Requirements](#requirements)
- [Task 1: Self-supervised and Supervised Learning on Image Classification](#task-1-self-supervised-and-supervised-learning-on-image-classification)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Task 2: CNN vs. Transformer for Image Classification](#task-2-cnn-vs-transformer-for-image-classification)
  - [Data Preparation](#data-preparation-1)
  - [Preliminaries](#preliminaries)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation-1)
  - [Visualization](#visualization-1)
- [Task 3: NeRF: 3D Reconstruction and View Synthesis](#task-3-nerf-3d-reconstruction-and-view-synthesis)
- [Directory Structure](#directory-structure)

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
tensorboard --logdir=./runs/
```

Then open the browser and go to `http://localhost:6006/`.

## Task 2: CNN vs. Transformer for Image Classification

### Data Preparation

Download the PASCAL VOC dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) for 2007 and [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) for 2012 into the `Midterm` directory.

### Preliminaries

Open the `VOC_MMDetection.ipynb` notebook, run the cell `Install MMDetection` to install the MMDetection toolkit (you can omit the clone step if you have cloned our repository).

Run the cell `Load the VOC dataset` to reorganize the dataset.

### Model Training

Run the following commands to train the Faster-RCNN or YOLOV3 model:

```cmd
!python mmdetection_voc/tools/train.py mmdetection_voc/configs/my_configs/VOC_faster-rcnn.py
!python mmdetection_voctools/train.py mmdetection_voc/configs/my_configs/VOC_yolov3.py
```

### Evaluation

First download the weights for the two models from [here](https://pan.baidu.com/s/1O-toY96MSuXnaVT4yOSMvQ?pwd=92wa) and put it in the `mmdetection_voc/work_dirs/VOC_faster-rcnn/` or `mmdetection_voc/work_dirs/VOC_yolov3/` directory.

Then open the `VOC_MMDetection.ipynb` notebook, run all the cells of `Inference`.

### Visualization

The visualization of the model performance can be found in the `figs` directory.

Run the following command to start the TensorBoard:

```cmd
tensorboard --logdir=./mmdetection_voc/work_dirs/VOC_faster-rcnn/20240602_143856
tensorboard --logdir=./mmdetection_voc/work_dirs/VOC_yolov3/20240602_160030
```

Then open the browser and go to `http://localhost:6006/`.

## Task 3: NeRF: 3D Reconstruction and View Synthesis


## Directory Structure

If you clone the repository and download all the required datasets and weights, the directory structure should look like this:
