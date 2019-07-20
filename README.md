## Overview

This assignment is to work out a system detecting cars in a video. To collect data, you can mount a camera to the hood(meaning the front) of the car, which takes pictures of the road every few seconds while you drive around. After gathering all these images in to a folder, we label them by drawing bounding boxes around every car we found. In this project, we use the data from [drive.ai](https://www.drive.ai/).
<!--more-->
repo: https://github.com/TianchaoHuo/Autonomous-driving-application-using-YOLO
## objective
We are target to detect the cars and build up bounding boxes for them. The bounding box should like as:
<div align=center><img src ="Autonomous-driving-application-using-YOLO\definition of a box.jpg"></div>

## dataset
1. Training set is from :[drive.ai](https://www.drive.ai/)
2. YOLO pre-trained neural network:
  - [YOLO weights](http://pjreddie.com/media/files/yolo.weights)
  - [YOLO cfg](https://github.com/pjreddie/darknet/tree/master/cfg)
  - [YAD2K](https://github.com/allanzelener/YAD2K)


## Methodology
Here I just briefly summary Object Detection delivered by Andrew Ng to make the logic of this assignment clearly.
### sliding windows
As for object detection, we firstly think about how to classify objects, meaning that input an image and output what is it. And then when we waner detect the object, localizing where is the object, we require the neural network to output the object coordinates, such as the centeral point and so on. One approach to do that is **sliding windows**.
<div align=center><img src ="Autonomous-driving-application-using-YOLO\sliding windows.jpg"></div>  

As shown above figure, we initially have a red window in the left-most cornner and then crop this corresponding size of image, inputing the neural network, to work out if here has the interested object. Repeating this process by shifting this window, we can finally traverse all the snippets of the image. However, this method has an obvious drawback, which is the computation cost is high. Therefore, we move to **convolutional sliding windows**.

## convolutional implementation of sliding windows

In order to implement  convolutional sliding windows, we need to know how to change from  fully conneted layers to convolutional layers of a neural network.

<div align=center><img src ="Autonomous-driving-application-using-YOLO\Turning FC layer into convolutional layers.jpg"></div>  

What we do is, as shown in above figure, to using 400 5x5 filters in fully conneted layer 1 so that we can have 1x1x400 output layer, instead of a vector of size 400. Similarly we use 400 1x1 filters and 4 1x1 filters in later fully conneted layers.
Next, we implement the sliding windows through convolution. Suppose that we have an input image with 16x16x3 and we use 5x5x3 windows to traverse this input image and then we can obtain the 12x12x16 ouput image after a convolutional process. Focusing on the step highlited by the red 4 superscript, we have four small grids, rather than only one, since we combine all the output into a big grid, representing the first convolutional output of the upper left area, the upper right, the  lower left and the lower right, separately. So, this principle of convolutional operation is that we do not need to divide the input image into 4 subsets, passing them through the forward propagation respectively.Instead we take this as an input image to feed to the neural network, as  2x2x4 shown here.

<div align=center><img src ="Autonomous-driving-application-using-YOLO\convolution implementation of sliding windows.jpg"></div>  

But it still has a drawback like the bounding box is not perfectly accurate.

### Bounding box predictions
Sliding windows method has a drawback, so-called disordered bounding boxes.
<div align=center><img src ="Autonomous-driving-application-using-YOLO\disordered bboxes.jpg"></div>  

You might see that the sliding winodw can not perfectly localize the car, or even we can say that the box (number [1]) is the most perfect box in this case. To address this problem, we introduce the YOLO algortihm, also meaning that You only look once [1]. Supposing we have an input image 100x100. and then we set a 3x3 net grid for simplicity.


<div align=center><img src ="Autonomous-driving-application-using-YOLO\Yolo algorithm.jpg"></div>  

we need to define a training set label such as shown in the above. $p_c$ denotes the probability having a interested object. $b_x,b_y,b_h,b_w$ represent the bounding box coordinate  and $c_1,c_2,c_3$ denote the probability of corresponding classes, like car, traffic lights, or pedestrian. For any one of this 9 grids, we can obtain a vector ouput of size 8x1, so that we can have 3x3x8 for total expected output size. The cental point of detected object will allow the object to be allocated into the grid in where the central point is, even though this object crosses through several grids. This works well if there is no more than one object in each grid. Besides, it also requires that the training data should have the same label as this.The advantages of YOLO algortihm is that outputs of a neural network include accurate bounding boxes.

### Intersection over union (Iou)

You might consider that the figure might has numerous bounding boxes for a single object as these snippets of a image are considered as interested objects by the sliding windows method, resulting redundant bounding boxes for a object. Therefore, Iou is set to measure the overlap between two bounding boxes.

<div align=center><img src ="Autonomous-driving-application-using-YOLO\IOU1.jpg"><img src ="Autonomous-driving-application-using-YOLO\IOU2.jpg"></div>
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\IOU3.jpg"></div>  

Generally we would say the detection is correct if $Iou > 0.5$ or alternatively you might choose $Iou > 0.6$ is OK.
Next, I waner introduce the non-max suppression algorithm.

###  non-max suppression algorithm
The algorithm may detect the same object multiple times.Non-maximum suppression is a way to make sure that your algorithm only detects each object once. Let's do an example.
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\NON-MAX.jpg"></div>  
It firstly selects the bounding box with the highest probability, in this case in 0.9 which is the most reliable detection, and then we highlite this bounding box. Next we compute the Iou between this highlited bounding box and the remaining bounding boxes  in sequence. By setting a threshold value for Iou, all the bounding boxes will be supressed if it has high Iou with the highlited bounding box. And then, the left bounding boxes will be assessed one by one to find the highest probability, in this case in 0.8. So that we can detect another car on the left and then do the non-max suppression algorithm again to omit the high Iou bounding boxes. Therefore, we can obtain the final result. This is basic concept of the non-max supression algorithm.

The following figure is to detail the algorithm:
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\nms.jpg"></div>  
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\nms1.jpg"></div>  
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\nms2.jpg"></div>



### anchor box
One of the problems with object detection so far is that each grid can only detect one object and if you want to one grid detects multiple objects, you can do this by using the concept of anchor box.
<div align=center ><img src ="Autonomous-driving-application-using-YOLO\anchor box.jpg"></div>
You can enlarge the vector to fill the different object parameters with anchor box. For example, the first 8 parameters of y vector is for the anchor box1, which is denoting the humanbeing and the remaining 8 parameters is denoting the car, which is also anchor box2. Now each object is assigned to the same grid as before. People would arbitrarily select the shape of anchor box and you can choose five to ten shapes of anchor box.



## Implementation
Basically I waner give some general workflow of this system.
1. Define anchor, classes, image shape. These come from "yolo_anchors.txt", "coco_classes.txt" and the image shape is (720,1280).
2. Loading the pre-trained model "Yolo.h5" file.
- We firstly download the "[yolo.weights](http://pjreddie.com/media/files/yolo.weights)", and then "[yolov2.cfg](https://github.com/pjreddie/darknet/tree/master/cfg)" and "[YAD2K](https://github.com/allanzelener/YAD2K)".
- put "yolo.weights", "yolo.cfg"(change name from yolov2.cfg to yolo.cfg) and "yad2k.py" into a floder and copy the model_data floader into this new floader.
- copy the yad2k floader in your download YAD2K-master into this new floader.
- avtivate your python environment that includes tensorflow
- excute "python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5"
you can refer to this [blog](https://blog.csdn.net/Solo95/article/details/85262828).
3. Obtain the outputs from the pre-trained model Yolo
4. Prediction
  - feed an input image to Yolo model
  - get the Yolo outputs (scores, boxes, classes)
  - yolo box to corners
  - yolo filter boxes
  - scale bboxes
  - non-max-suppression
  - showing the result

**Summary for YOLO**: - Input image (608, 608, 3) - The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. - After flattening the last two dimensions, the output is a volume of shape (19, 19, 425): - Each cell in a 19x19 grid over the input image gives 425 numbers. - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. - 85 = 5 + 80 where 5 is because  (𝑝𝑐,𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤)  has 5 numbers, and and 80 is the number of classes we'd like to detect - You then select only few boxes based on: - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes - This gives you YOLO's final output.


## Result
We firstly and simply use one image to test the model as shown in the left below. And then the model will output the right below image drawing several bounding boxes and corresponding probability and classes.
<div><img width=550 height=330  src ="Autonomous-driving-application-using-YOLO\test.jpg"><img width=550 height=330   src ="Autonomous-driving-application-using-YOLO\test_result.jpg"></div>

And the visual demo you can go to this blog:
https://tianchaohuo.github.io/2019/07/17/Autonomous-driving-application-using-YOLO/
</br>



## Conclusions
This is simply impplement the Yolo algorithm. For the future work, it is expected to work with videos in real-time.
**What you should remember**: - YOLO is a state-of-the-art object detection model that is fast and accurate - It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. - The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes. - You filter through all the boxes using non-max suppression. Specifically: - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes - Intersection over Union (IoU) thresholding to eliminate overlapping boxes - Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.

## References

[1] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)</br>
[2] Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)</br>
[3] Allan Zelener - YAD2K: Yet Another Darknet 2 Keras</br>
[4] The official YOLO website (https://pjreddie.com/darknet/yolo/)</br>
[5] How to build Yolo.h5, https://blog.csdn.net/Solo95/article/details/85262828</br>
[6] Andrew Ng, https://mooc.study.163.com/term/2001392030.htm
