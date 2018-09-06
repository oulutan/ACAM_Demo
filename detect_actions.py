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
    video_path = "./tests/chase1Person1View3Point0.mp4"

    main_folder = './'
    # obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')

    print("Loading object detection model at %s" % obj_detection_graph)

    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker()

    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn')
    ckpt_name = 'model_ckpt_RGB_soft_attn-9'
    input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()
    
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)

    reader = imageio.get_reader(video_path, 'ffmpeg')

    frame_cnt = 0
    for test_img in reader:
        frame_cnt += 1
        if frame_cnt % 2 == 0:
            continue
        print("frame_cnt: %i" %frame_cnt)
        # Object Detection
        expanded_img = np.expand_dims(test_img, axis=0)
        detection_list = obj_detector.detect_objects_in_np(expanded_img)
        detection_info = [info[0] for info in detection_list]
        tracker.update_tracker(detection_info, test_img)

        # Action detection
        batch_size = len(tracker.active_actors)
        batch_np = np.zeros([batch_size, act_detector.timesteps] + act_detector.input_size + [3])
        rois_np = np.zeros([batch_size, 4])
        batch_indices_np = np.array(range(batch_size))
        for bb, actor_info in enumerate(tracker.active_actors):
            actor_no = actor_info['actor_id']
            tube, roi = tracker.crop_person_tube(actor_no)
            batch_np[bb, :] = tube
            rois_np[bb]= roi
        feed_dict = {input_seq:batch_np, rois:rois_np, roi_batch_indices:batch_indices_np}
        probs = act_detector.session.run(pred_probs, feed_dict=feed_dict)

        print_top_k = 5
        for bb in range(batch_size):
            act_probs = probs[bb]
            order = np.argsort(act_probs)[::-1]
            print("Person %i" % tracker.active_actors[bb]['actor_id'])
            for pp in range(print_top_k):
                print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))


if __name__ == '__main__':
    main()
