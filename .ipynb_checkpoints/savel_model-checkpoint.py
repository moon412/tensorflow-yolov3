#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : save_model.py
#   Author      : Yue Zhao
#   Created date: 2019-08-18 
#   Description : This script loads the Yolo V3 model and 
#                 loads the ckpt file into an active session. 
#                 Save the model # in SaveModel format for TF serving
#
#================================================================


import tensorflow as tf
from core.yolov3 import YOLOV3
from tensorflow.saved_model import signature_def_utils, signature_constants, tag_constants
from tensorflow.saved_model import utils as save_model_utils

savemodel_file_path = "./savemodel/yolo_v3/1"

ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2",
                     "pred_multi_scale/concat"]

img_size = 416
num_channels = 3
with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, num_channels], name='input_data')

model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
print("{} trainable variables".format(len(tf.trainable_variables())))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    
    x_op = sess.graph.get_operation_by_name('input/input_data')
    x = x_op.outputs[0]
    pred_op = sess.graph.get_operation_by_name('pred_multi_scale/concat')
    pred = pred_op.outputs[0]
        
    print("prediction signature")
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={"input": save_model_utils.build_tensor_info(x)},
        outputs={"output":save_model_utils.build_tensor_info(pred)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
        
    builder = tf.saved_model.builder.SavedModelBuilder(savemodel_file_path)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], 
                                         signature_def_map={"predict": prediction_signature})
        
    print("save...")
    builder.save()

sess.close()


