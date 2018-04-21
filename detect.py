import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import scipy.io
import scipy.misc
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Input, Lambda
from keras.models import Model, load_model
from matplotlib.pyplot import imshow

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_boxes_to_corners, yolo_head,
                                     yolo_loss)
from yolo_utils import (draw_boxes, generate_colors, preprocess_image,
                        read_anchors, read_classes, scale_boxes)

IMG_SZ = [416., 416., 3.]

class YOLO:

  def __init__(self):
    self.model = load_model("model_data/yolo.h5")
    self.class_names = read_classes("model_data/coco_classes.txt")
    self.anchors = read_anchors("model_data/yolo_anchors.txt")
    self._create_placeholders()
    self._evaluate_output()
    self.scores, self.boxes, self.classes = self._yolo_eval(self.outputs,
                                                            (720., 1280.))

  def _create_placeholders(self):
    with tf.name_scope('placeholders'):
      self.X = tf.placeholder(tf.float32, shape=[None, *IMG_SZ], name='X')

  def _evaluate_output(self):
    with tf.name_scope('output'):
     self.outputs = yolo_head(self.model.output, self.anchors,
                              len(self.class_names))

  def _yolo_filter_boxes(self, box_confidence, boxes, box_class_probs,
                          threshold=.6):
    '''
    box_confidence : bs x 19 x 19 x anchors x 1
    boxes          : bs x 19 x 19 x anchors x 4
    box_class_probs: bs x 19 x 19 x anchors x #classes
    threshold      : float
    '''
    # bs x 19 x 19 x anchors x #classes
    box_scores = box_confidence * box_class_probs

    # bs x 19 x 19 x anchors
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # bs x 19 x 19 x anchors
    mask = box_class_scores > threshold

    scores = tf.boolean_mask(box_class_scores, mask)
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_classes, mask)

    return scores, boxes, classes

  def _yolo_non_max_suppression(self, scores, boxes, classes, max_boxes=10,
                                iou_threshold=0.5):

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes,
                                               iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

  def _yolo_eval(self, yolo_outputs, image_shape, max_boxes=10,
                 score_threshold=0.6, iou_threshold=0.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = self._yolo_filter_boxes(box_confidence, boxes,
                                                     box_class_probs,
                                                     score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = self._yolo_non_max_suppression(scores, boxes,
                                                            classes, max_boxes,
                                                            iou_threshold)

    return scores, boxes, classes

def predict():
  sess = K.get_session()

  yolo = YOLO()

  image_file = 'test.jpg'
  image, image_data = preprocess_image("images/" + image_file,
                      model_image_size=list(map(lambda x: int(x), IMG_SZ[:2])))


  out_scores, out_boxes, out_classes = sess.run(
                  [yolo.scores, yolo.boxes, yolo.classes],
                  feed_dict={yolo.model.input:image_data, K.learning_phase():0})

  print('Found {} boxes for {}'.format(len(out_boxes), image_file))
  colors = generate_colors(yolo.class_names)
  draw_boxes(image, out_scores, out_boxes, out_classes, yolo.class_names,
             colors)
  image.save(os.path.join("out", image_file), quality=90)
  output_image = scipy.misc.imread(os.path.join("out", image_file))
  imshow(output_image)

if __name__ == '__main__':
  predict()
