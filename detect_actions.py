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
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, required=True)

    args = parser.parse_args()

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "./output_videos/%s_output.mp4" % basename

    # video_path = "./tests/chase1Person1View3Point0.mp4"
    # out_vid_path = 'output.mp4'

    main_folder = './'
    ## Best
    # obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    ## Good and Faster
    obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    ## Fastest


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

    print("Reading video file %s" % video_path)
    reader = imageio.get_reader(video_path, 'ffmpeg')
    fps_divider = 2
    fps = reader.get_meta_data()['fps'] // fps_divider
    writer = imageio.get_writer(out_vid_path, fps=fps)
    print("Writing output to %s" % out_vid_path)

    frame_cnt = 0
    for cur_img in reader:
        frame_cnt += 1
        if frame_cnt % fps_divider != 0:
            continue
        print("frame_cnt: %i" %frame_cnt)
        # Object Detection
        expanded_img = np.expand_dims(cur_img, axis=0)
        detection_list = obj_detector.detect_objects_in_np(expanded_img)
        detection_info = [info[0] for info in detection_list]
        tracker.update_tracker(detection_info, cur_img)

        # Action detection
        no_actors = len(tracker.active_actors)
        batch_np = np.zeros([no_actors, act_detector.timesteps] + act_detector.input_size + [3])
        rois_np = np.zeros([no_actors, 4])
        batch_indices_np = np.array(range(no_actors))
        for bb, actor_info in enumerate(tracker.active_actors):
            actor_no = actor_info['actor_id']
            tube, roi = tracker.crop_person_tube(actor_no)
            batch_np[bb, :] = tube
            rois_np[bb]= roi
        if tracker.active_actors:
            max_batch_size = 10
            prob_list = []
            cur_index = 0
            while cur_index < no_actors:
                cur_batch = batch_np[cur_index:cur_index+max_batch_size]
                cur_roi = rois_np[cur_index:cur_index+max_batch_size]
                cur_indices = batch_indices_np[cur_index:cur_index+max_batch_size]
                feed_dict = {input_seq:cur_batch, rois:cur_roi, roi_batch_indices:cur_indices}
                cur_probs = act_detector.session.run(pred_probs, feed_dict=feed_dict)
                prob_list.append(cur_probs)
                cur_index += max_batch_size
            probs = np.concatenate(prob_list, axis=0)

        # Print top_k probs
        print_top_k = 5
        act_results = []
        for bb in range(no_actors):
            act_probs = probs[bb]
            order = np.argsort(act_probs)[::-1]
            print("Person %i" % tracker.active_actors[bb]['actor_id'])
            cur_results = []
            for pp in range(print_top_k):
                print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            act_results.append(cur_results)
        out_img = visualize_detection_results(cur_img, tracker.active_actors, act_results, display=False)
        writer.append_data(out_img)
        
    writer.close()


np.random.seed(10)
COLORS = np.random.randint(0, 255, [300, 3])
def visualize_detection_results(img_np, active_actors, act_results, display=True):
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        cur_act_results = act_results[ii]
        cur_box, cur_score, cur_class = cur_actor['all_boxes'][-1], cur_actor['all_scores'][-1], 1
        actor_id = cur_actor['actor_id']
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box


        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        #label = bbox['class_str']
        # label = 'Class_%i' % cur_class
        label = obj.OBJECT_STRINGS[cur_class]['name']
        message = '%s_%i: %% %.2f' % (label, actor_id,conf)
        action_message_list = ["%s:%.3f" % (actres[0][0:5], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        # action_message = " ".join(action_message_list)

        color = COLORS[actor_id]

        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (255,0,0), 1)

    if display: 
        cv2.imshow('results', disp_img)
        cv2.waitKey(0)
    return disp_img

if __name__ == '__main__':
    main()
