# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



The Project
---

The goal is to write a software pipeline to detect vehicles in a video.

This project is based on YOLO (V2) papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242).

[YOLO](https://pjreddie.com/darknet/yolo/) ("you only look once") is a state-of-the-art, real-time object detection system. 


This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

## Installation
**This env. requires keras version 2.1.5**

Create env using conda as `conda env create -f environment.yml`

##### Pretrained YOLO v2 weights


To convert pre-trained YOLO weights to keras model -
* Download YOLO weights from http://pjreddie.com/media/files/yolo.weights and put in project root
* Run `python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5`

YAD2k - https://github.com/allanzelener/YAD2K
yolo.cfg - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg

To install OpenCV for python 3.6 and window - download OpenCV wheel from [OpenCV](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) and run pip install as
`pip install opencv_python-3.4.1+contrib-cp36-cp36m-win_amd64.whl`

For more detail refer - [Installing OpenCV](https://stackoverflow.com/questions/42994813/installing-opencv-on-windows-10-with-python-3-6-and-anaconda-3-6)

