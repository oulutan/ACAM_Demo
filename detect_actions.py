import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
import argparse

import object_detection.object_detector as obj
import action_detection.action_detector as act

def main():
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-v', '--video_path', type=str, required=True)

    main_folder = './'
    # obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')

    print("Loading object detection model at %s" % obj_detection_graph)

    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker()
    sess = obj_detector.session

    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn', session=sess)
    ckpt_name = 'model_ckpt_RGB_soft_attn-9'
    input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    act_detector.restore_model(ckpt_path)

if __name__ == '__main__':
    main()
