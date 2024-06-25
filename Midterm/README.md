<h1 align="center">DATA130051.01 Midterm Project</h1>
<h3 align="center"> Wu Jia'ao  21307130203 </h3>

## Contents
- [Contents](#contents)
- [Requirements](#requirements)
- [Task 1: Fine-tuning ResNet18 on CUB-200-2011](#task-1-fine-tuning-resnet18-on-cub-200-2011)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Task 2: Object Detection on PASCAL VOC](#task-2-object-detection-on-pascal-voc)
  - [Data Preparation](#data-preparation-1)
  - [Preliminaries](#preliminaries)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation-1)
  - [Visualization](#visualization-1)
- [Directory Structure](#directory-structure)

***

## Requirements
This project requires Python >= 3.8. See the requirements.txt file for the required packages. You can install them using the following command:

```cmd
pip install -r requirements.txt
```

> [!NOTE]
> You need to open the `Midterm` directory as the working directory to run the scripts and notebooks correctly.

## Task 1: Fine-tuning ResNet18 on CUB-200-2011

### Data Preparation
Download the CUB-200-2011 dataset from [here](https://data.caltech.edu/records/65de6-vp158) into the `Midterm` directory.

run the following command to reorganize the dataset:

```cmd
python CUB_load.py
```

### Training the Model
Open the `CUB_finetune.ipynb` notebook, run all the cells before `Functions` (included) and run the cell `Fine-tuning the model` or `Training from scratch`.

### Evaluation
First download the weights for the finetuned model from [here](https://pan.baidu.com/s/1O-toY96MSuXnaVT4yOSMvQ?pwd=92wa) and put it in the `Midterm` directory.

Then open the `CUB_finetune.ipynb` notebook, run all the cells before `Functions` (included) and run the cell `Evaluation`.

### Visualization
The visualization of the model performance can be found in the `figs` directory.

Run the following command to start the TensorBoard:

```cmd
# Choose one of the following
tensorboard --logdir=./runs/CUB_bird_classification
tensorboard --logdir=./runs/CUB_from_scratch
```

Then open the browser and go to `http://localhost:6006/`.

## Task 2: Object Detection on PASCAL VOC

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

## Directory Structure
If you clone the repository and download all the required datasets and weights, the directory structure should look like this:

```
Midterm
├─ CUB_best_weights_ft.pth           Weights for the finetuned ResNet18 model (Task 1)
├─ CUB_best_weights_scratch.pth      Weights for the scratch ResNet18 model (Task 1)
├─ CUB_finetune.ipynb                Notebook for finetuning the ResNet18 model (Task 1)
├─ CUB_load.py                       Script for loading snd reorganizing the CUB dataset (Task 1)
├─ CUB_200_2001                      CUB Dataset (Task 1)
├─ figs                              Directory for storing figures
│  ├─ external_comparison.png        Comparison of model performance on external images (Task 2)
│  ├─ faster-rcnn                    Faster-RCNN figures (Task 2)
│  │  ├─ comparison.png              Comparison of proposal and final boxes (Task 2)
│  │  ├─ external/vis                Faster-RCNN bounding boxes on external images (Task 2)
│  │  ├─ final/vis                   Final Faster-RCNN bounding boxes on VOC test images (Task 2)
│  │  └─ proposal                    Proposal boxes for Faster-RCNN on VOC test images (Task 2)
│  ├─ TensorBoard                    All TensorBoard visualizations
│  ├─ yolov3/vis                     YOLOV3 bounding boxes on external images (Task 2)
├─ mmdetection_voc                   Toolkit for object detection (Task 2)
│  ├─ configs                        Configuration files for Faster-RCNN and YOLOV3
│  │  ├─ my_configs                  Custom configurations for Faster-RCNN and YOLOV3
│  │  │  ├─ VOC_faster-rcnn.py
│  │  │  └─ VOC_yolov3.py
│  │  └─ _base_                      Base configurations for Faster-RCNN and YOLOV3 (from MMDetection)
│  │     ├─ datasets
│  │     │  └─ voc0712.py
│  │     ├─ default_runtime.py
│  │     └─ models
│  │        └─ faster-rcnn_r50_fpn.py
│  ├─ data/VOCdevkit                 VOC dataset
│  │  ├─ VOC2007
│  │  └─ VOC2012
│  ├─ demo                           Demo images for object detection
│  │  ├─ external                    External images outside the VOC dataset (3 images)
│  │  └─ from_VOCtest-2007           VOC test images (4 images)
│  ├─ tools                          Tools for training and testing the models (from MMDetection)
│  │  ├─ test.py
│  │  └─ train.py
│  └─ work_dirs                      Directory for storing the weights and logs
│     ├─ VOC_faster-rcnn
│     │  ├─ faster_rcnn.pth          Weights for the Faster-RCNN model
│     │  └─ VOC_faster-rcnn.py       Script for training the Faster-RCNN model (from MMDetection)
│     └─ VOC_yolov3
│        ├─ VOC_yolov3.pth           Weights for the YOLOV3 model
│        └─ VOC_yolov3.py            Script for training the YOLOV3 model (from MMDetection)
├─ runs                              TensorBoard logs (Task 1)
│  ├─ CUB_bird_classification
│  └─ CUB_from_scratch
└─ VOC_MMDetection.ipynb             Notebook for object detection on PASCAL VOC (Task 2)
```