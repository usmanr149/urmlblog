---
layout: post
title: Implementing a Single Shot Detector Model in Tensorflow 2.0
categories: [Computer Vision]
tags: [Anchor Boxes, Computer Vision, SSD]
---

Single Shot MultiBox Detector (SSD) detects objects in images using a single deep neural network. In this blog post I will cover how to implement SSD300 (meaning images are 300x300) detector presented <a href="https://arxiv.org/pdf/1512.02325.pdf" target="_blank">here</a> in Tensorflow 2.0.

# What do we Need?

SSD needs an input image with ground truth boxes for each objects. In addition we also need default boxes (also called anchor boxes), for each default box we predict the offset and the confidence score for all object categories.



At training time, the default boxes are matched to the ground truth boxes. The matched boxes are treated as positives and the rest of the default boxes are treated as negative. The model loss is a weighted sum between localization loss and confidence loss.


## Model Configuration

The SSD approach produces a fixed-size collection of default boxes. The model then predict the offset and classification scores for in the default boxes. The early network laters are based on standard architecture used for image classification (VVG16, CGG16, MobileNetV1, MobileNetV2, etc.). Auxilary structure is added to the base model to product the offset and classification scores for the default boxes.  In this blog I will be going over how to use VGG16 as the base model layer and building a SSD model off off it.

![_config.yml]({{ site.baseurl }}/images/SSD/VGG-16-SSD.png)

SSD uses multi-scale feature maps for detections at multiple scales. We generate default boxes that is associated with each feature map cell. For example in the image below the CNN, we can see how the CNN output maps to the default boxes.

![_config.yml]({{ site.baseurl }}/images/SSD/Default_Boxes_and_CNN_outputs.png)

For each default box we generate $(4 + c + 1)$ outputs, where $c$ is the number of classes in the dataset and +1 for background.

## Matching Default Boxes to Ground Truth Boxes

To train the model we need to determine which default boxes correspond to ground truth detection. Each ground truth box is matched to the default box with the highest Intersect over Union (IOU). We then also match default boxes to any ground truth boxes with IOU > threshold (set as 0.5).

## Training Objectives

The loss function for the SSD model is the weighted sum of the localization loss and the confidence loss:

$$
L(x, c, l, g) = \dfrac{1}{N}(L_{conf}(x,x) + \alpha L_{loc}(x,l,g))
$$

where 
$N$ is the total number of matched default boxes, edge case: if N = 0 then set loss to 0, 
$x$ is an indicator for matching the $i$-th default boxes to the $j$-th ground truth box, 
$l$ is the predicted box, 
$g$ is the ground truth box
$c$ is the predicted class.

For the localization loss we regress to offsets for the $(cx, cy, w, h)$ of the matching default bounding boxes. See Figure ~ for a description of $cx, cy, w$ and  $h$.

$$
L_{loc}(x,l,g) = \sum^N_{i \in Pos} \sum_{m \in \{cx, cy,w,h \}}smooth_{\text{L1}}(l_i^m - \hat{g}_j^m)
$$

where

$$
\hat{g}_j^{cx} = 5 * \dfrac{g_j^{cx} - d_i^{cx}}{d_i^w} \\
\hat{g}_j^{cy} = 5 * \dfrac{g_j^{cy} - d_i^{cy}}{d_i^h} \\
\hat{g}_j^{w} = 10 * \text{log}\left(\dfrac{g_j^w}{d_i^w}\right) \\
\hat{g}_j^{h} = 10 * \text{log}\left(\dfrac{g_j^h}{d_i^h}\right)
$$

The prior variances or the scaling constants are not mentioned in the paper, but they are used in the code, I read about them <a href="https://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html" target="_blank">here</a>. The author of the paper suggests in a github comment that they can also be understood as loss smoothing factors, the confidence and the localization loss are on a different scale and using the scaling constants brings these two losses together allowing for a smoother loss function that is easy to optimize.

## Implementing SSD in Tensorflow 2.0

Now that we have an understanding of the theory behind SSD model architecture, we can now implement the model in Tensorflow 2.0. I have already covered how to generate anchor boxes in a previous post so I will not go over them here. We 

