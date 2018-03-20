
**Vehicle Detection Project**

The goal is to write a software pipeline to detect vehicles in a video

[//]: # (Image References)
[image1]: ./output_images/test1.jpg
[image2]: ./output_images/test2.jpg
[image3]: ./output_images/test3.jpg
[image4]: ./output_images/test4.jpg
[image5]: ./output_images/test5.jpg
[image6]: ./output_images/test6.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Rubric points are covered here. Specific installation requirements are mentioned in README.md

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This project is based on YOLO (V2) papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242). 

See YOLO Model below for detailed explanation.


#### 2. Explain how you settled on your final choice of HOG parameters.

See YOLO Model below for detailed explanation.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

See YOLO Model below for detailed explanation.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

See YOLO Model below for detailed explanation.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Test image example

**Image 1** - correctly predicts 2 cars with confidence 0.74 and 0.73
![alt text][image1]

**Image 2** - correctly predicts no image
![alt text][image2]

**Image 3** - incorrectly predicts no car
![alt text][image3]

**Image 4** - correctly predicts 2 cars with confidence 0.78 and 0.72
![alt text][image4]

**Image 5** - correctly predicts 2 cars with confidence 0.84 and 0.85
![alt text][image5]

**Image 6** - correctly predicts 2 cars with confidence 0.80 and 0.73
![alt text][image6]
---

### YOLO

YOLO ("you only look once") is a state-of-the-art, real-time object detection system.

This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

#### Model Details
* Input - batch of images of shape (m, 608, 608, 3)
* Output - list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  (pc,bx,by,bh,bw,c)

..* pc - class confidence
..* bx,by - center of the box relative to the bounds of the grid cell
..* bh,bw - width and height are predicted relative to the whole image
..* c - vector of 80 class

Each bounding box can be thought of represented by 85 numbers. Network outputs prediction for 5 predefined bounding boxes (also called anchor boxes)

YOLO architecture  -> IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85)

#### Model Architecture

| Layer (type)                   | Output Shape       | Param #     | Connected to              |
| ------------------------------ |:------------------:| :----------:| -------------------------:|
|input_1 (InputLayer)            |(None, 608, 608, 3) | 0           |                           |
|conv2d_1 (Conv2D)               |(None, 608, 608, 32)| 864         |input_1[0][0]              |
|batch_normalization_1 (BatchNorm| (None, 608, 608, 32)|  128       |conv2d_1[0][0]             |      
|leaky_re_lu_1 (LeakyReLU)        |(None, 608, 608, 32)  |0          | batch_normalization_1[0][0]      |
|max_pooling2d_1 (MaxPooling2D)   |(None, 304, 304, 32)  |0          | leaky_re_lu_1[0][0]              |
|conv2d_2 (Conv2D)                |(None, 304, 304, 64)  |18432      | max_pooling2d_1[0][0]            |
|batch_normalization_2 (BatchNorm |(None, 304, 304, 64)  |256        | conv2d_2[0][0]                   |
|leaky_re_lu_2 (LeakyReLU)        |(None, 304, 304, 64)  |0          | batch_normalization_2[0][0]      |
|max_pooling2d_2 (MaxPooling2D)   |(None, 152, 152, 64)  |0          | leaky_re_lu_2[0][0]              |
|conv2d_3 (Conv2D)                |(None, 152, 152, 128) |73728      | max_pooling2d_2[0][0]            |
|batch_normalization_3 (BatchNorm |(None, 152, 152, 128) |512        | conv2d_3[0][0]                   |
|leaky_re_lu_3 (LeakyReLU)        |(None, 152, 152, 128) |0          | batch_normalization_3[0][0]      |
|conv2d_4 (Conv2D)                |(None, 152, 152, 64)  |8192       | leaky_re_lu_3[0][0]              |
|batch_normalization_4 (BatchNorm |(None, 152, 152, 64)  |256        | conv2d_4[0][0]                   |
|leaky_re_lu_4 (LeakyReLU)        |(None, 152, 152, 64)  |0          | batch_normalization_4[0][0]      |
|conv2d_5 (Conv2D)                |(None, 152, 152, 128) |73728      | leaky_re_lu_4[0][0]              |
|batch_normalization_5 (BatchNorm |(None, 152, 152, 128) |512        | conv2d_5[0][0]                   |
|leaky_re_lu_5 (LeakyReLU)        |(None, 152, 152, 128) |0          | batch_normalization_5[0][0]      |
|max_pooling2d_3 (MaxPooling2D)   |(None, 76, 76, 128)   |0          | leaky_re_lu_5[0][0]              |
|conv2d_6 (Conv2D)                |(None, 76, 76, 256)   |294912     | max_pooling2d_3[0][0]            |
|batch_normalization_6 (BatchNorm |(None, 76, 76, 256)   |1024       | conv2d_6[0][0]                   |
|leaky_re_lu_6 (LeakyReLU)        |(None, 76, 76, 256)   |0          | batch_normalization_6[0][0]      |
|conv2d_7 (Conv2D)                |(None, 76, 76, 128)   |32768      | leaky_re_lu_6[0][0]              |
|batch_normalization_7 (BatchNorm |(None, 76, 76, 128)   |512        | conv2d_7[0][0]                   |
|leaky_re_lu_7 (LeakyReLU)        |(None, 76, 76, 128)   |0          | batch_normalization_7[0][0]      |
|conv2d_8 (Conv2D)                |(None, 76, 76, 256)   |294912     | leaky_re_lu_7[0][0]              |
|batch_normalization_8 (BatchNorm |(None, 76, 76, 256)   |1024       | conv2d_8[0][0]                   |
|leaky_re_lu_8 (LeakyReLU)        |(None, 76, 76, 256)   |0          | batch_normalization_8[0][0]      |
|max_pooling2d_4 (MaxPooling2D)   |(None, 38, 38, 256)   |0          | leaky_re_lu_8[0][0]              |
|conv2d_9 (Conv2D)                |(None, 38, 38, 512)   |1179648    | max_pooling2d_4[0][0]            |
|batch_normalization_9 (BatchNorm |(None, 38, 38, 512)   |2048       | conv2d_9[0][0]                   |
|leaky_re_lu_9 (LeakyReLU)        |(None, 38, 38, 512)   |0          | batch_normalization_9[0][0]      |
|conv2d_10 (Conv2D)               |(None, 38, 38, 256)   |131072     | leaky_re_lu_9[0][0]              |
|batch_normalization_10 (BatchNor |(None, 38, 38, 256)   |1024       | conv2d_10[0][0]                  |
|leaky_re_lu_10 (LeakyReLU)       |(None, 38, 38, 256)   |0          | batch_normalization_10[0][0]     |
|conv2d_11 (Conv2D)               |(None, 38, 38, 512)   |1179648    | leaky_re_lu_10[0][0]             |
|batch_normalization_11 (BatchNor |(None, 38, 38, 512)   |2048       | conv2d_11[0][0]                  |
|leaky_re_lu_11 (LeakyReLU)       |(None, 38, 38, 512)   |0          | batch_normalization_11[0][0]     |
|conv2d_12 (Conv2D)               |(None, 38, 38, 256)   |131072     | leaky_re_lu_11[0][0]             |
|batch_normalization_12 (BatchNor |(None, 38, 38, 256)   |1024       | conv2d_12[0][0]                  |
|leaky_re_lu_12 (LeakyReLU)       |(None, 38, 38, 256)   |0          | batch_normalization_12[0][0]     |
|conv2d_13 (Conv2D)               |(None, 38, 38, 512)   |1179648    | leaky_re_lu_12[0][0]             |
|batch_normalization_13 (BatchNor |(None, 38, 38, 512)   |2048       | conv2d_13[0][0]                  |
|leaky_re_lu_13 (LeakyReLU)       |(None, 38, 38, 512)   |0          | batch_normalization_13[0][0]     |
|max_pooling2d_5 (MaxPooling2D)   |(None, 19, 19, 512)   |0          | leaky_re_lu_13[0][0]             |
|conv2d_14 (Conv2D)               |(None, 19, 19, 1024)  |4718592    | max_pooling2d_5[0][0]            |
|batch_normalization_14 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_14[0][0]                  |
|leaky_re_lu_14 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_14[0][0]     |
|conv2d_15 (Conv2D)               |(None, 19, 19, 512)   |524288     | leaky_re_lu_14[0][0]             |
|batch_normalization_15 (BatchNor |(None, 19, 19, 512)   |2048       | conv2d_15[0][0]                  |
|leaky_re_lu_15 (LeakyReLU)       |(None, 19, 19, 512)   |0          | batch_normalization_15[0][0]     |
|conv2d_16 (Conv2D)               |(None, 19, 19, 1024)  |4718592    | leaky_re_lu_15[0][0]             |
|batch_normalization_16 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_16[0][0]                  |
|leaky_re_lu_16 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_16[0][0]     |
|conv2d_17 (Conv2D)               |(None, 19, 19, 512)   |524288     | leaky_re_lu_16[0][0]             |
|batch_normalization_17 (BatchNor |(None, 19, 19, 512)   |2048       | conv2d_17[0][0]                  |
|leaky_re_lu_17 (LeakyReLU)       |(None, 19, 19, 512)   |0          | batch_normalization_17[0][0]     |
|conv2d_18 (Conv2D)               |(None, 19, 19, 1024)  |4718592    | leaky_re_lu_17[0][0]             |
|batch_normalization_18 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_18[0][0]                  |
|leaky_re_lu_18 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_18[0][0]     |
|conv2d_19 (Conv2D)               |(None, 19, 19, 1024)  |9437184    | leaky_re_lu_18[0][0]             |
|batch_normalization_19 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_19[0][0]                  |
|conv2d_21 (Conv2D)               |(None, 38, 38, 64)    |32768      | leaky_re_lu_13[0][0]             |
|leaky_re_lu_19 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_19[0][0]     |
|batch_normalization_21 (BatchNor |(None, 38, 38, 64)    |256        | conv2d_21[0][0]                  |
|conv2d_20 (Conv2D)               |(None, 19, 19, 1024)  |9437184    | leaky_re_lu_19[0][0]             |
|leaky_re_lu_21 (LeakyReLU)       |(None, 38, 38, 64)    |0          | batch_normalization_21[0][0]     |
|batch_normalization_20 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_20[0][0]                  |
|space_to_depth_x2 (Lambda)       |(None, 19, 19, 256)   |0          | leaky_re_lu_21[0][0]             |
|leaky_re_lu_20 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_20[0][0]     |
|concatenate_1 (Concatenate)      |(None, 19, 19, 1280)  |0          | space_to_depth_x2[0][0]          |
|                                 |                      |           | leaky_re_lu_20[0][0]             |
|conv2d_22 (Conv2D)               |(None, 19, 19, 1024)  |11796480   | concatenate_1[0][0]              |
|batch_normalization_22 (BatchNor |(None, 19, 19, 1024)  |4096       | conv2d_22[0][0]                  |
|leaky_re_lu_22 (LeakyReLU)       |(None, 19, 19, 1024)  |0          | batch_normalization_22[0][0]     |
|conv2d_23 (Conv2D)               |(None, 19, 19, 425)   |435625     | leaky_re_lu_22[0][0]             |

-Total params: 50,983,561
-Trainable params: 50,962,889
-Non-trainable params: 20,672

#### Implementation
---
#### Pre-processing
---

##### Resizing
Size of input image was 1280 * 720. This image was re-sized to 608 * 608 as model input is 608 *608

code -> method `preprocess_image` of `utils.py`

##### Normalization
Each image pixel is normalized to have values between 0 and 1 by diving each pixel with 255

code -> method `preprocess_image` of `utils.py`
 
##### Training 
I have used YOLO2 pre-trained weight. To convert weights to keras model I have used Allan Zelener [YAD2K](https://github.com/allanzelener/YAD2K) 

I have also used some of the utility method from YAD2k (as module yad2k.models)

To convert pre-trained YOLO weights to keras model -
* Download YOLO weights from http://pjreddie.com/media/files/yolo.weights and put in project root
* Run `python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5`

YAD2k - https://github.com/allanzelener/YAD2K
yolo.cfg - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg


#### Post Processing
---
YOLO2 divides each image in 19 * 19 cells and predicts image if center of image falls in that grid cell.
There are 5 anchor boxes associated with each 19 * 19 cell and each cell outputs 85 values (pc,bx,by,bh,bw,c) 

###### Filter on class threshold 
Filter all boxes whose class scores less than threshold (0.6). This method is implemented `filter_anchor_boxes` in `yolo.py`

###### Non max suppression (NMS)
Even after filtering by thresholding over the classes scores, there are still lots of overlapping boxes. 
A second filter for selecting the right boxes is called non-maximum suppression (NMS)
Non-max suppression uses "Intersection over Union", or IoU, which is measure of overlap between 2 bounding boxes. 
I have used IoU threshold of 0.5 (`non_max_suppression` of `yolo.py`)

##### Drawing the bounding boxes
The predictions (bx, by) for each bounding box are relative to the bounds of the grid cell and (bw, bh) are relative to the whole image. 
To compute the final bounding box we need to scale predicted boxes back to the original image

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
---
I have used YOLO2

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used YOLO2 which has close to 51 million parameter. Running this on project video on 8 core cpu machine takes close to 2 hours.
Tiny YOLO would have been better choice.    

