[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 













## Training, Predicting and Scoring ##

### Architecture

#### Fully Convolutional Networks

**Convolution Neural Networks(CNN)** tells what objects are there in an image, but it doesn’t localize the object in the 
image i.e it doesn’t tell where the object lies in the image. 
The Fully Convolutional Networks(FCN) is built to be able to localize the object of a class in the image. 
The FCN segment’s the object within the image. 
Fully convolutional networks are capable of this via a process called semantic segmentation. 

**Semantic Segmentation** is a three step process which classifies, localize’s and segments the object.

1. The first step is classification, which consists of making a prediction for a whole input.

2. The second step is localization/detection, which provide not only the classes but also additional information regarding 
the spatial location of those classes.

3. The third and final step is, making dense predictions inferring labels for every pixel, so that each pixel is labeled 
with the class of its enclosing object region.

The FNC contains Encoder section, 1x1 convolution layer and a decoder section.

**Encoder:**

The encoder is usually a pre-trained classification network like VGG/ResNet which classifies the image. 
The encoder network performs convolution with filter bank to produce a set of feature maps. 
More the layers in encoder network, better the refined feature map for classifying the object, 
but having several layers can achieve more translation invariance for robust classification. 
Also there is a loss of spatial resolution of the feature maps.

**1x1 Convolution Layer:**

The 1x1 convolution is a regular convolution, with a kernel and stride of 1. 
Using a 1x1 convolution layer allows the network to be able to retain spatial information from the encoder. 
The 1x1 convolution layers allows the data to be both flattened for classification while retaining spatial information.

**Decoder:**

The decoder network upsamples its input feature maps using the memorized max-pooling indices from the corresponding encoder 
feature maps producing sparse feature maps. The decoder model can either be composed of transposed convolution layers or 
bilinear upsampling layers.

The transposed convolution layer is the inverse of regular convolution layers, multiplying each pixel of the input with the kernel. 
Where bilinear upsampling is similar to ‘Max Pooling’ and uses the weighted average of the four nearest pixels from the given pixel, 
estimating the new pixel intensity value.


**Skip Connections:**

The FCN also uses the skip connections which is skipping some layers in the neural network and feeding the output of one 
layer to another layer skipping few layers in between. Usually, some information is captured in initial layers and is required 
for reconstruction during the up-sampling of the features. By skipping the layers we can feed the information from primary 
layers to the later layers resulting in better spatial information and segmentation.

**Model Architecture Used:**

Layers|Type|Dimension
:------:|:----:|:---------:
layer_1|input_layer|160x160x3
layer_2|Encoder_1|80x80x32
layer_3|Encoder_2|40x40x64
layer_4|1x1 Conv|40x40x128
layer_5|Decoder_1|80x80x64
layer_6|Decoder_2|160x160x32
layer_7|Conv layer|160x160x3


The FCN model used for the project contains two encoder layers, 1 1x1 concolution layers and 2 decoder layers.

The first convolution uses the filter of size 32 and a stride of 2, while the second convolution uses a filter size of 
64 and a stride of 2. Both encoder layers use ‘SAME’ padding. Then finally encoding using1x1 convolution layer with filter size of 
128 and kernel and stride of size of 1.

The first decoder layer uses the 1x1 convolution layer’s output as the small input layer and the first convolution layer as 
large input layer, thus mimicking a skip connection. The decoder uses the filter of size 64, where the second decoder layer which 
has filter size of 32 uses the output of first decoder layer as small input  layer and the original image as large input layer 
thus persorming another skip connection to retain the information better through the network. 

The output convolution layer applies softmax function to the output of the second decoder layer thus classifying each pixel 
and segmenting the object.

**Hyperparameters:**

Name|Value
:----:|:---------:
learning rate | 0.0015
batch_size | 64
epochs | 50
steps per epoch | 200
validation steps | 50
workers | 2

These are the optimal hyperparameters that i found for my network by manual tuning and trail & error. 

**Training:**

The model was trained using Nvidia Geforce GTX 1080 with 12GB of RAM.

**Performance:**

The final score of my model is 0.3569
The final IoU is 0.4921

**Future Enhancement:**

The code performance can be further enhanced by implementing a learning rate decay to undertake a performance-based search.

The network can be further enhanced by increasing the layers in encoder and decoder section.

The network can also be trained to identify other objects along with person.
