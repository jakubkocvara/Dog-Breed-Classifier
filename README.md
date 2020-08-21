# Dog Breed Classifier
![](readme_pic.png)

## Project Overview
This is the capstone project of the Udacity data science nanodegree. 
You can read the original README and the instructions on how to run this project here: [https://github.com/udacity/dog-project](https://github.com/udacity/dog-project).

We are training Convolutional Neural Networks to identify dog-breeds based on an image. First one from scratch and in the second part we are using transfer learning using pre-trained models.

## Building our own CNN
After consulting existing models for similar purposes I settled on an architecture of alternating convolution and maxpool layers, finishing with two fully connected layers like this: 

<img src="cnn_scratch.png" width="300" />

With this architecture after 30 epochs we are achieving around 8% prediction accuracy.

## Creating a CNN by transfer learning
<img src="cnn_xception.png" width="300" />
