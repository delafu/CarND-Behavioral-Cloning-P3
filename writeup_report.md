# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "Model Visualization"
[image2]: ./images/center_2017_07_12_08_26_19_702.jpg "Center"
[image3]: ./images/center_2017_07_12_08_29_11_488.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 a video recording of your vehicle driving autonomously at least one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I modified the drive.py to crop the images the before to input them to the model. The modifications are documented in the code in (drive.py lines 64-65)
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia architecture mentionated in the lesson

![alt text][image1]

I´ve done tests with the LeNet acrhitecture too. But I decided to use the Nvidia one because I´ve obtained better results but surprisingly I was able to finish one lap using LeNet.

The model consists of 5 convolutional layers. The first three layers use a 5x5 kernel and a stride of 2x2. The output depth of these three layers are 24, 36 and 48. The last two layers use a 3x3 kernel and a stride of 1x1. The ouput depth of these two layers is 6.

Next I use a Flatten layer and I apply a Droput layer with a value of 0.9 to try to reduce overfiting. I´ve found that when I decrease the Dropout value the car drives worse.

The next three are fully connected layers with 100, 50, 10 units and finally the output with one unit because we are in a regression problem.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The model definition is in (model.py lines 119-131)

#### 2. Attempts to reduce overfitting in the model

The model contain a dropout layer in order to reduce overfitting (model.py lines 127). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded multiple laps that I did with the sim. I tried to drive the most of the laps to drive in the center of the road and some in the left and in the right of the road but it was very difficult.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the lessons of the web. 

First In begin with only one fully connected layer like in the lesson and then I was making the model more complex. I used a LeNet network and worked on preprocessing the images and selecting the corrections in the steer angle measurements in the left and right cameras

The LeNet model worked well but did not work well all the laps and I decided to try the Nvidia model. 

I have to increase the correction to 0.3 and -0.3 from 0.2 because the car went out of the road.

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 119-131) is the Nvidia architecture plus one Dropout layer.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I recorded multiple laps that I did with the sim. I tried to drive the most of the laps to drive in the center of the road and some in the left and in the right of the road but it was very difficult.

![alt text][image3]


I didn´t take samples on the track two. I rode the car several times but I was not able to finish two laps and record them.

To augment the data set, I also flipped images and angles and I think It´s a very good idea to get more left curves. I did some tests not augmenting the images with steering angle of 0 but I got better results augmenting them

I´m cropping all the images to only train the network with the lower side of the images (model.py lines 29-30 56-57)

Although I´ve implmented the generator I´m not using it in my last models because I have a GTX 1080 graphic card and 32 GB RAM computer and the generator slows down the training velocity drastically.

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I´ve found that a good number of epochs is 5. Sometimes when I try 3 th car goes out off the road.

I think that the val_loss in this exercise is not very important because I´ve done some test that obtained a a low val_loss and the car worked worse than in another test with a greater val_loss.
