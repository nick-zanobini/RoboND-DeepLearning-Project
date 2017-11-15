[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

[image_0]: ./docs/misc/sim_screenshot.png
[image_1]: ./docs/misc/network_arch.png
[image_2]: ./docs/misc/CUDA8_Install.png
[image_3]: ./docs/misc/NVIDIA_Drivers.png
[image_4]: ./docs/misc/Verified_Driver.png


## Deep Learning Project ##

I trained the model at home on my GTX-980 to follow the hero based on the provided training set. I have included the steps to install TensorFlow GPU and CUDA8 on Ubuntu 16.04.


## Network Architecture
The final fully connected network consists of 2 encoder layers and two decoder layers connected by a 1x1 convolution layer  The  1x1 convolution layer is necessary to retain spacial information from each image. Normally simply a fully connected network would be used to classify images but a network with fully connected layers would not retain any spacial information hence why every layer is a convolutional layer.

The encoding stage is used to extract features for segmentation so simple features can be recognized early on and the deeper the network goes the more complex features it can recognize. The skip connections help retain some of the original information that would normally be lost. The decoder stage upscales the output of the encoder stage to the size of the original image. Lastly the convolutional output layer with softmax activation makes the pixel-wise comparison between the three classes.

![alt text][image_1] 
## Training
After reading others experiences on Slack I decided to simply install TensorFlow with GPU support CUDA 8 and cuDNN. I tested several different setups to fully utilize the 4 GB of RAM on my GPU and ended up with 100 epochs, 150 steps per epochs, a batch size of 24 and 4 workers and was training each step in around 90 seconds. I would've preferred to use a higher batch size but 32 was all the memory my GPU had. To figure this out I started with a batch size of 64, 4 workers, 100 epochs and 100 steps per epoch at a learning rate of 0.005. This setup quickly ran out of memory and I continually adjusted my batch size down until it was able to train the model. After I settled on a batch size I continued to tweak the other parameters like epochs and steps per epoch to maximize my GPU usage. I would train a model and if it didn't run out of memory I would just increase the number of epochs or the number of steps per epoch until i was able to train the model without running out of memmory. Just reading the comments and suggestions in the Slack channel I chose a learning rate of 0.005 which yielded impressive results and didn't feel the need to tweak it that much, I tried +/- .002 on the learning rate and didn't notice a change so I stuck with my inital value. The validation steps I left at the deualt value of 50 because it seemed to be good enough. 8 workers always seemed to be a quick way to run out of memory as well as using anything over 24 as a batch size.


## Results
My final score was 43% (.43) and an IOU score of 55% but this score might be a little over inflated as it was tested using the provided data. This network architecture could be used to track a cat or a dog or any other image assuming it was trained to said image. It would be interesting if was identifying everything it found in each from and then the user could decide which detected object the drone should follow. 

## Future Enhancements
I think it could be interesting to add two more encoder and decoder layers and expirament with what is the optimum number of encoder / decoder layers. At what point do you start to overtrain the model. I would to test that out and find the limit exirametnally. It wouldnt be that hard to do, if I was able to generate a huge dataset and train it on a system running a NVIDIA TitanXp or two running SLI and a batch size of 256 or 512 and a couple hundred epochs, I think I could find the limit fairly quickly. I think generating a huge dataset and training a network like aformentioned would yield impressive results as well. In simulation it's ok for us to have a 40% score but in real life, follow me mode would require much higher accuracy. Assuming I could train a model on a system like this and prove I could achieve results that would be considered practical to use in a public place I think it would be amazing to migrate this project from the world of simulation to a custom drone that I have. That would require someone filming me walking around and generating another huge data set with video from my drone. 

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

**Setup Computer for Home Training**  
* Prepare Computer
    ```
    sudo apt-get update
    sudo apt-get upgrade  
    sudo apt-get install build-essential cmake g++ gfortran 
    sudo apt-get git pkg-config python-dev 
    sudo apt-get software-properties-common wget
    sudo apt-get autoremove 
    sudo rm -rf /var/lib/apt/lists/*
    sudo apt-get install cuda -y
    ```

* Install NVIDIA Graphics Card Drivers
    ```
    lspci | grep -i nvidia
    ```
    * Which should yield:
        ![alt text][image_3] 
    
    * Next add the proprietary repository of NVIDIA drivers
        ```
        sudo add-apt-repository ppa:graphics-drivers/ppa
        sudo apt-get update
        sudo apt-get install nvidia-398
        ```
    
    * Once NVIDIA driver is installed, restart the computer. You can verify the driver using the following command.
        ```
        cat /proc/driver/nvidia/version
        ```
        ![alt text][image_4] 
    

* Install NVIDIA Drivers  
  * [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)  
  * [cuDNN v7.0.3 for CUDA 8.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.3/prod/8.0_20170926/cudnn-8.0-linux-x64-v7-tgz)  
    ![alt text][image_2]
    
* Install cuDNN
	```
	tar -xzvf cudnn-8.0-linux-x64-v7.tgz
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

	```

* Link CUDA 8 to CUDA 5 for Tensorflow
	```
    cd /usr/local/cuda/lib64/
    sudo ln -s libcudnn.so.5.1.10 libcudnn.so.5
    sudo ln -s libcudnn.so.5 libcudnn.so

	```

* Check the links
	```
	ls -l libcudnn*
	```
	* Should give:
        ```
        -rwxr-xr-x 1 root root 217188104 Nov  6 22:07 libcudnn.so
        lrwxrwxrwx 1 root root        17 Nov  6 22:21 libcudnn.so.5 -> libcudnn.so.7.0.3
        -rwxr-xr-x 1 root root 217188104 Nov  6 22:07 libcudnn.so.7
        -rwxr-xr-x 1 root root 217188104 Nov  6 22:07 libcudnn.so.7.0.3
        -rw-r--r-- 1 root root 211053738 Nov  6 22:07 libcudnn_static.a
        ```

* Copy `cudnn.h` in the `include` directory to `usr/local/cuda/include`
	```
	sudo cp cudnn.h /usr/local/cuda/include/
	```
	
* Append to `~/.bashrc`:
    ```
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda
    ```
    
* Reload `~/.bashrc`:
    ```
    source ~/.bashrc
    ```
	
* Install TensorFlow-GPU
    ```
    sudo pip install tensorflow-gpu==1.2.1
    ```

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

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a tempory folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaluate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`
