---
layout: post
title: Implementing a Single Shot Detector Model in Tensorflow 2.0
categories: [Computer Vision]
tags: [Anchor Boxes, Computer Vision, SSD]
---

Single Shot MultiBox Detector (SSD) detects objects in images using a single deep neural network. In this blog post I will cover how to implement SSD300 (meaning images are scaled to 300x300 before being fed to the model) detector presented <a href="https://arxiv.org/pdf/1512.02325.pdf" target="_blank">here</a> in Tensorflow 2.0.

# What do we Need?

To train an SSD model, we need an input image with ground truth boxes for each objects. The figure below shows an example of an image with a ground truth label for aeroplane:

![_config.yml]({{ site.baseurl }}/images/SSD/image_w_bbox.png)

In addition we also need default boxes (also called anchor boxes). At training time, the default boxes are matched to the ground truth boxes. The image below shows all of the default boxes that overlap with the ground truth box.

![_config.yml]({{ site.baseurl }}/images/SSD/default_boxes_w_IOU_g_0.5.png)

The matched boxes are treated as positives and the rest of the default boxes are treated as negative. 

## Model Configuration

The SSD approach produces a fixed-size collection of default boxes. The model then predict the offset and classification scores for each of the default boxes. The early network layers are based on standard architecture used for image classification (<a href="https://keras.io/api/applications/vgg/" target="_blank">VGG16</a>, <a href="https://keras.io/api/applications/vgg/" target="_blank">VGG19</a>, <a href="https://keras.io/api/applications/mobilenet/" target="_blank">MobileNetV1</a>, <a href="https://keras.io/api/applications/mobilenet/" target="_blank">MobileNetV2</a>, etc.). Auxilary structure is added to the base model to produce the offset and classification scores for the default boxes.  In this blog I will be going over how to use VGG16 as the base model layer and build a SSD model off off it.

![_config.yml]({{ site.baseurl }}/images/SSD/VGG-16-SSD.png)

SSD uses multi-scale feature maps for detections at multiple scales. We generate default boxes that are associated with each feature map cell. For example in the image below, we can see how the CNN output maps to the default boxes for an image.

![_config.yml]({{ site.baseurl }}/images/SSD/Default_Boxes_and_CNN_outputs.png)

For each default box we generate $(4 + c + 1)$ outputs, where 4 because of the offsets, $c$ is the number of classes in the dataset and +1 for background.

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
\hat{g}_j^{h} = 10 * \text{log}\left(\dfrac{g_j^h}{d_i^h}\right) \label{eq:offsets}
$$

The prior variances or the scaling constants are not mentioned in the paper, but they are used in the code, I read about them <a href="https://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html" target="_blank">here</a>. The author of the paper suggests in a github comment that they can also be understood as loss smoothing factors, the confidence and the localization loss are on a different scale and using the scaling constants brings these two losses together allowing for a smoother loss function that is easy to optimize.

## Implementing SSD in Tensorflow 2.0

Now that we have an understanding of the theory behind SSD model architecture, we can now implement the model in Tensorflow 2.0. I have already covered how to generate anchor boxes in a previous post so I will not go over them here. Once we have the default boxes we can then match these default boxes with the ground truth boxes. One thing to note is that the method to generate default boxes covered in my previous blogs outputs the boxes in centroid represtation: $[cx, cy, w, h]$, to calculate IOUs between two rectangles it is best to represent to use the the corners representation: $[xmin, ymin, xmax, ymax]$.

![_config.yml]({{ site.baseurl }}/images/SSD/Centroid Representation.jpg)

The code below contains helper functions to generate default bounding boxes, find default boxes for which IOU > threshold with ground truth boxes, calculate offsets for training objectives as shown in Eq. \ref{eq:offsets}


```python
def calculate_scale_of_default_boxes(k, m, s_max = 0.9, s_min = 0.2):
    """
    m = number_of_feature_maps
    s_k = s_min + (s_max - s_min) * (k - 1)/(m - 1)
    width_k = s_k * sqrt(aspect_ratio)
    height_k = s_k / sqrt(aspect_ratio)
    """
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)

def generate_default_boxes(feature_map_shapes, number_of_feature_maps, aspect_ratios):
    """
    feature map shapes for VGG: [38, 19, 10, 5, 3, 1]
    """

    assert len(feature_map_shapes) == number_of_feature_maps, 'number of feature maps needs to be {0}'.format(len(feature_map_shapes))
    assert len(feature_map_shapes) == len(aspect_ratios), 'Need aspect ratios for all feature maps'

    prior_boxes = []

    for k, f_k in enumerate(feature_map_shapes):
        s_k = calculate_scale_of_default_boxes(k, m = number_of_feature_maps)
        s_k_prime = np.sqrt(s_k * calculate_scale_of_default_boxes(k + 1, m = 6))
        for i in range(f_k):
            for j in range(f_k):
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                prior_boxes.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    # height, width for numpy
                    prior_boxes.append([cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)])

    prior_boxes = tf.convert_to_tensor(prior_boxes, dtype=tf.float32)
    return tf.clip_by_value(prior_boxes, clip_value_min = 0., clip_value_max = 1.)


# Adapted from https://gist.github.com/escuccim/d0be49ccfc6084cdc784a67339f130dd
def box_overlap_iou(boxes, gt_boxes):
    """
    Args:
        boxes: shape (total boxes, x_min, y_min, x_max, y_max)
        gt_boxes: shape (1, total label, x_min  y_min, x_max, y_max)

    Returns:
        Tensor with shape (batch_size, total boxes, total label)
    """
    box_x_min, box_y_min, box_x_max, box_y_max = tf.split(boxes, 4, axis = 1)
    gt_boxes_x_min, gt_boxes_y_min, gt_boxes_x_max, gt_boxes_y_max = tf.split(gt_boxes, 4, axis = 2)

    # From https://www.tensorflow.org/api_docs/python/tf/transpose
    intersection_x_min = tf.maximum(box_x_min, tf.transpose(gt_boxes_x_min, perm=[0, 2, 1]))
    intersection_y_min = tf.maximum(box_y_min, tf.transpose(gt_boxes_y_min, perm=[0, 2, 1]))

    intersection_x_max = tf.minimum(box_x_max, tf.transpose(gt_boxes_x_max, perm=[0, 2, 1]))
    intersection_y_max = tf.minimum(box_y_max, tf.transpose(gt_boxes_y_max, perm=[0, 2, 1]))

    # need to take care of boxes that don't overlap at all
    intersection_area = tf.maximum(intersection_x_max - intersection_x_min, 0) * tf.maximum(intersection_y_max - intersection_y_min, 0)

    boxes_areas = (box_x_max - box_x_min) * (box_y_max - box_y_min)
    gt_box_areas = (gt_boxes_x_max - gt_boxes_x_min) * (gt_boxes_y_max - gt_boxes_y_min)

    union = (boxes_areas + tf.transpose(gt_box_areas, perm=[0, 2, 1])) - intersection_area

    return tf.maximum(intersection_area / union, 0)


def match_priors_with_gt(prior_boxes, boxes, gt_boxes, gt_labels, number_of_labels, threshold = 0.5):
    
    """
    prior boxes: (1, number of default boxes, c_x, c_y, w, h)
    boxes: shape (total boxes, x_min, y_min, x_max, y_max)
    gt_boxes: (1, number of labels, x_min, y_min, x_max, y_max)
    gt_labels: (1, 1 label per each gt box)

    0 is background, so the gt_labels is the number of labels in the dataset + 1
    class 0 is reserved.
    """

    # number of rows for the IOU map the is the number of gt_boxes
    IOU_map = box_overlap_iou(boxes, gt_boxes)

    # convert ground boxes labels to box label format
    gt_box_label = convert_to_centre_dimensions_form(gt_boxes)

    # select the box with the highest IOU
    highest_overlap_idx = tf.math.argmax(IOU_map, axis = 1)
    highest_overlap_idx = tf.cast(highest_overlap_idx, tf.int32)
    idx = tf.range(IOU_map.shape[1])
    highest_overlap_idx_map = tf.expand_dims(tf.equal(idx, tf.transpose(highest_overlap_idx)), axis = 0)
    IOU_map = tf.where(tf.transpose(highest_overlap_idx_map, perm=[0,2,1]), tf.constant(1.0), IOU_map)

    # find the column idx with the highest IOU at each row
    max_IOU_idx_per_row = tf.math.argmax(IOU_map, axis = 2)
    # find the max value per row
    max_IOU_per_row = tf.reduce_max(IOU_map, axis = 2)

    # threshold IOU
    max_IOU_above_threshold = tf.greater(max_IOU_per_row, threshold)
    
    # map the gt boxes to the prior boxes with the highest overlap
    gt_box_label_map = tf.gather(gt_box_label, max_IOU_idx_per_row, batch_dims = 1)
    # get the offset, offcet (delta_cx, delta_cy, delta_width, delta_height)
    gt_box_label_map_offsets = calculate_offset_from_gt(gt_box_label_map, prior_boxes)
    # remove from gt_boxes_map where overlap with prior boxes is less than 0.5
    gt_boxes_map_offset_suppressed = tf.where( tf.expand_dims(max_IOU_above_threshold, -1),  
                                        gt_box_label_map_offsets, tf.zeros_like(gt_box_label_map))
    # add a positive condition column for the localization loss
    max_IOU_above_threshold_expand = tf.expand_dims(max_IOU_above_threshold, -1)
    max_IOU_above_threshold_expand = tf.cast(max_IOU_above_threshold_expand, tf.float32)
    gt_boxes_map_offset_suppressed_with_pos_cond = tf.concat([  gt_boxes_map_offset_suppressed, 
                                                                max_IOU_above_threshold_expand ], axis = 2)


    gt_labels_map = tf.gather(gt_labels, max_IOU_idx_per_row, batch_dims = 1)
    # suppress the label where IOU with the gt boxes is < 0.5
    gt_labels_map_suppressed = tf.where( max_IOU_above_threshold, 
                                        gt_labels_map, tf.zeros_like(gt_labels_map))
    gt_labels_one_hot_encoded = tf.one_hot(gt_labels_map_suppressed, number_of_labels)

    return gt_boxes_map_offset_suppressed_with_pos_cond, gt_labels_one_hot_encoded

def calculate_offset_from_gt(gt_boxes_mapped_to_prior, prior_boxes):
    prior_boxes = tf.expand_dims(prior_boxes, axis=0)
    g_j_cx = 10 * (gt_boxes_mapped_to_prior[:, :, 0] - prior_boxes[:, :, 0]) / prior_boxes[:, :, 2]
    g_j_cy = 10 * (gt_boxes_mapped_to_prior[:, :, 1] - prior_boxes[:, :, 1]) / prior_boxes[:, :, 3]
    g_j_w = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 2] / prior_boxes[:, :, 2])
    g_j_h = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 3] / prior_boxes[:, :, 3])

    offset = tf.concat( [ g_j_cx, g_j_cy, g_j_w, g_j_h ] , axis = 0)

    return tf.transpose(tf.expand_dims(offset, axis = 0), perm=[0,2,1])

def convert_to_box_form(boxes):
    """
    Input:
        (number_of_labels, c_x, c_y, width, height)
    Output:
        (number_of_labels, x_min, y_min, x_max, y_max)
    """

    box_coordinates = tf.concat([   boxes[:, :2] - boxes[:, 2:] / 2, 
                                    boxes[:, :2] + boxes[:, 2:] / 2 ], 
                                    axis = 1)

    return tf.clip_by_value(box_coordinates, clip_value_min = 0., clip_value_max = 1.)

def convert_to_centre_dimensions_form(boxes):
    """
    Input:
        boxes: (1, number_of_labels, x_min, y_min, x_max, y_max)
    Output:
        (1, number_of_labels, c_x, c_y, width, height)
    """

    coordinates = tf.concat([
                [
                        (boxes[:, :, 0] + boxes[:, :, 2]) / 2., 
                        (boxes[:, :, 1] + boxes[:, :, 3]) / 2.,
                        boxes[:, :, 2] - boxes[:, :, 0],
                        boxes[:, :, 3] - boxes[:, :, 1]
                ]], axis = 1)
    # need the output in the same format as input, could be imporived
    coordinates = tf.transpose(coordinates, perm=[1,2,0])
    return tf.clip_by_value(coordinates, clip_value_min = 0., clip_value_max = 1.)
```

## Implementing SSD Loss

