---
layout: post
title: Generating Anchor Boxes
categories: [Computer Vision]
tags: [Anchor Boxes, Computer Vision]
---

# Anchor Boxes

Anchor boxes are predifined collection of boxes into which the ground truth labels will be mapped. The anchor boxes encompass a variety widths, heights and aspect ratios to match the possible ground truth label for objects in a dataset. For the VGG16-Single Shot Detector architecture 8732 anchor boxes are generated, Figure 1 shows just the 2% of all the anchor boxes generated for the VVG16-Single Shot Detector.

![_config.yml]({{ site.baseurl }}/images/SSD/SSD_VGG16_AnchorBoxes.png)

# Advantages of Anchor Boxes

The advantage of anchor boxes is that you can evaluate all object predictions at once. Anchor boxes eliminate the need for scanning an image or making multiple passes throungh the model for each area of interest.

# How to Create Anchor Boxes?

The shape and number of anchor boxes generated depends on the underlying Convolution Neural Network architecture used for mapping, the number of features selected and the number of aspect ratio per feature. Here we will will take a look at VGG16 network and discuss how it's architecture is used to determine the number and sizes of the the anchorbox. Figure 2 shows what the VGG16 architecture for classification looks like:

![_config.yml]({{ site.baseurl }}/images/SSD/VGG16-architecture.jpeg)
*From <a href="https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/" target="_blank">Hands-on Transfer Learning with Keras and the VGG16 Model</a> by James McDermott. The VGG16 model architecture for a classification process.*

In SSD archtecture, we remove the ending fully connected layers:

![_config.yml]({{ site.baseurl }}/images/SSD/VGG16-architecture-remove-fcnn.jpeg)
*From <a href="https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/" target="_blank">Hands-on Transfer Learning with Keras and the VGG16 Model</a> by James McDermott. The last last layer are removed from VGG16 when it is tranformed to an SSD.*

and use the output of the conv4_3 layer as a feature map.

![_config.yml]({{ site.baseurl }}/images/SSD/SSD_Architecture1.png)
*Use the output from the conv4_3 layer as a feature map on which to base detections.*

But thats not all, in SSD we want to make detections at various scales, we want to detect large objects small objects and everything in-between. So we add layers to generate a stack of feature maps with a variety of sizes. These feature maps will be used to make detections.

![_config.yml]({{ site.baseurl }}/images/SSD/SSD_Architecture2.png)
*Feature maps generated from the base VGG16 model.*

The SSD paper explains that these feature maps will have different receptive field sizes, the anchor boxes we will generate don't need to exactly corresond to the actual receptive fields. The paper presents a method to generate anchor "boxes so that specific feature maps learn to be responsive to particular scales of the objects."

Generating these anchorboxes is relatively straightforward. The SSD paper presents a simple way to genrate the anchorboxes using the following equation:

$$
s_{k} = s_{\text{min}} + \dfrac{(s_{\text{max}} - s_{\text{min}})}{m - 1} (k - 1),~~~~~~ k \in [0, m)
$$

where $s_{\text{min}}$ = 0.2, $s_{\text{max}}$ = 0.9, $k$ is the feature map number and $m$ is the total number of feature maps, for VGG16 it is 6 (see Figure 3).

For now lets ignore the aspect ratios, in that case the width and height of the anchor box is $s_{k}$, the center of the anchor box is given by 

$$
\left(\dfrac{i + 0.5}{f_k}, \dfrac{j + 0.5}{f_k} \right)
$$

where $i,j \in [0, f_k)$

For VGG16, $f_k = (38, 19, 10, 5, 3, 1)$

We can generate the anchor boxes using the following code snippet:

```python
anchor_boxes = []
k = 0
f_0 = 38

s_min = 0.2
s_max = 0.9
m = 6

s_1 = s_min + (s_max - s_min) * (k - 1) / (m - 1)

for i in range(f_1):
    for j in range(f_1):
        cx = (i + 0.5) / float(f_k)
        cy = (j + 0.5) / float(f_k)
        anchor_boxes.append([cx, cy, s_1, s_1])
```

On a 300x300 image, this what the anchor boxes look like:

![_config.yml]({{ site.baseurl }}/images/SSD/conv4_3_anchor_boxes.png)
*Anchor boxes that correspond to conv4_3 feature map from VGG16.*

This is not all though, as we can also add more anchor boxes by varying the aspect ratios. The SSD paper recommends using 5 aspect ratios denoted as $a_{r} \in \{1,2,3,1/2,1/3\}$. The width and height for each aspect ratio is altered by the following equations:

$$
w_{k}^{a} = s_{k}\sqrt(a_{r}) \\
h_{k}^{a} = s_{k}/\sqrt(a_{r})
$$

For the aspect ratio of 1, an additional default box whose scale is $s_{k'} = \sqrt{s_{k}s_{k+1}}$ is also added. This means that 6 default anchor boxes are generated per feature map location, so for $f_0$, we will get $38 \times 38 \times 6$ anchor boxes.

