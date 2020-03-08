---
layout: post
title: How to Rotate YOLO Bounding Boxes
---

![_config.yml]({{ site.baseurl }}/images/rotate_yolo_bbox/rotated_bbox.jpg)
*Rotate the image along with the bounding boxes.*

Image augmentation is a common technique in computer vision to increase the 
diversity of images present in a data set. One of the main challenges in 
computer vision is tagging, you only want to tag the original images and not 
the augmented images.

Recently while working on an image detection problem I wrote some code to rotate
YOLO mark labels to create new images.

YOLO mark is GUI for drawing bounding boxes of objects in images for YOLO v3 and v2 training.

For example I can use Yolo mark to draw bounding box around planes in this pictures.

![_config.yml]({{ site.baseurl }}/images/rotate_yolo_bbox/airplane_bbox_original.png)
*Airplanes*

Now I would like to rotate the image and the bounding boxes I generated using the 
Yolo_mark tool.

### How Does Yolo_mark Format Work?

If we draw the following bounding box using Yolo_mark,

![_config.yml]({{ site.baseurl }}/images/rotate_yolo_bbox/Yolo_bbox.jpg)
 
then a '.txt' file will automatically be created with the following line:

$$
\begin{align}
image label, center\_x/W, center\_y/H, bbox\_width/W, bbox\_height/H
\end{align}
$$

where $$W$$ is the image width and $$H$$ is the image height.

### How to Rotate the Yolo_mark Format Bounding Box?

