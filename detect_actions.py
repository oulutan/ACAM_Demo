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

import time
DISPLAY = False
SHOW_CAMS = False
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-d', '--display', type=str, required=False, default="True")

    args = parser.parse_args()
    display = (args.display == "True" or args.display == "true")
    
    #actor_to_display = 6 # for cams

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "./output_videos/%s_output.mp4" % (basename if not SHOW_CAMS else basename+'_cams_actor_%.2d' % actor_to_display)
    #out_vid_path = './output_videos/testing.mp4'

    # video_path = "./tests/chase1Person1View3Point0.mp4"
    # out_vid_path = 'output.mp4'

    main_folder = './'

    # NAS

    obj_detection_model =  'ssd_mobilenet_v2_coco_2018_03_29'
    obj_detection_graph = os.path.join("object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")



    print("Loading object detection model at %s" % obj_detection_graph)


    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker()

    


    print("Reading video file %s" % video_path)
    reader = imageio.get_reader(video_path, 'ffmpeg')
    action_freq = 8
    # fps_divider = 1
    print('Running actions every %i frame' % action_freq)
    fps = reader.get_meta_data()['fps'] #// fps_divider
    W, H = reader.get_meta_data()['size']
    T = tracker.timesteps
    if not display:
        writer = imageio.get_writer(out_vid_path, fps=fps)
        print("Writing output to %s" % out_vid_path)

    
    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn')
    #ckpt_name = 'model_ckpt_RGB_soft_attn-16'
    #ckpt_name = 'model_ckpt_soft_attn_ava-23'
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

    #input_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf([T,H,W,3])
    memory_size = act_detector.timesteps - action_freq
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory([T,H,W,3], memory_size)
    
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    

    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)

    prob_dict = {}
    frame_cnt = 0
    for cur_img in reader:
        frame_cnt += 1
        #tracker.add_frame(cur_img)
        print("frame_cnt: %i" %frame_cnt)
        # Object Detection
        expanded_img = np.expand_dims(cur_img, axis=0)
        #expanded_img = np.tile(expanded_img, [10,1,1,1]) # test the speed
        t1 = time.time()
        detection_list = obj_detector.detect_objects_in_np(expanded_img)
        detection_info = [info[0] for info in detection_list]
        t2 = time.time(); print('obj det %.2f seconds' % (t2-t1))
        tracker.update_tracker(detection_info, cur_img)
        t3 = time.time(); print('tracker %.2f seconds' % (t3-t2))
        no_actors = len(tracker.active_actors)

        if tracker.active_actors and frame_cnt % action_freq == 0:
            probs = []

            cur_input_sequence = np.expand_dims(np.stack(tracker.frame_history[-action_freq:], axis=0), axis=0)

            rois_np, temporal_rois_np = tracker.generate_all_rois()
            if no_actors > 14:
                no_actors = 14
                rois_np = rois_np[:14]
                temporal_rois_np = temporal_rois_np[:14]

            #feed_dict = {input_frames:cur_input_sequence, 
            feed_dict = {updated_frames:cur_input_sequence, # only update last #action_freq frames
                         temporal_rois: temporal_rois_np,
                         temporal_roi_batch_indices: np.zeros(no_actors),
                         rois:rois_np, 
                         roi_batch_indices:np.arange(no_actors)}
            run_dict = {'pred_probs': pred_probs}
            if SHOW_CAMS:
                run_dict['cropped_frames'] = cropped_frames
                #import pdb;pdb.set_trace()
                run_dict['final_i3d_feats'] =  act_detector.act_graph.get_collection('final_i3d_feats')[0]
                #run_dict['cls_weights'] = [var for var in tf.global_variables() if var.name == "CLS_Logits/kernel:0"][0]
                run_dict['cls_weights'] = act_detector.act_graph.get_collection('variables')[-2] # this is the kernel
            #import pdb;pdb.set_trace()
            out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
            probs = out_dict['pred_probs']
            # associate probs with actor ids
            print_top_k = 5
            for bb in range(no_actors):
                act_probs = probs[bb]
                order = np.argsort(act_probs)[::-1]
                cur_actor_id = tracker.active_actors[bb]['actor_id']
                print("Person %i" % cur_actor_id)
                cur_results = []
                for pp in range(print_top_k):
                    print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                    cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                prob_dict[cur_actor_id] = cur_results

            t5 = time.time(); print('action %.2f seconds' % (t5-t3))
        # # Action detection
        # no_actors = len(tracker.active_actors)
        # #batch_np = np.zeros([no_actors, act_detector.timesteps] + act_detector.input_size + [3], np.uint8)
        # batch_list = []
        # rois_np = np.zeros([no_actors, 4])
        # batch_indices_np = np.array(range(no_actors))
        # for bb, actor_info in enumerate(tracker.active_actors):
        #     actor_no = actor_info['actor_id']
        #     tube, roi = tracker.crop_person_tube(actor_no)
        #     #batch_np[bb, :] = tube
        #     batch_list.append(tube)
        #     rois_np[bb]= roi
        #t4 = time.time(); print('cropping %.2f seconds' % (t4-t3))

        # if tracker.active_actors:
        #     batch_np = np.stack(batch_list, axis=0)
        #     max_batch_size = 10
        #     prob_list = []
        #     cur_index = 0
        #     while cur_index < no_actors:
        #         cur_batch = batch_np[cur_index:cur_index+max_batch_size]
        #         cur_roi = rois_np[cur_index:cur_index+max_batch_size]
        #         cur_indices = batch_indices_np[cur_index:cur_index+max_batch_size] - cur_index
        #         feed_dict = {input_seq:cur_batch, rois:cur_roi, roi_batch_indices:cur_indices}
        #         #t51 = time.time(); print('action before run %.2f seconds' % (t51-t4))
        #         cur_probs = act_detector.session.run(pred_probs, feed_dict=feed_dict)
        #         #t52 = time.time(); print('action after run %.2f seconds' % (t52-t51))
        #         prob_list.append(cur_probs)
        #         cur_index += max_batch_size
        #     probs = np.concatenate(prob_list, axis=0)

        #t5 = time.time(); print('action %.2f seconds' % (t5-t4))
        # Print top_k probs
        #out_img = visualize_detection_results(cur_img, tracker.active_actors, prob_dict)
        if frame_cnt > 16:
            out_img = visualize_detection_results(tracker.frame_history[-16], tracker.active_actors, prob_dict)
            if SHOW_CAMS:
                if tracker.active_actors:
                    actor_indices = [ii for ii in range(no_actors) if tracker.active_actors[ii]['actor_id'] == actor_to_display]
                    if actor_indices:
                        out_img = visualize_cams(out_img, cur_input_sequence, out_dict, actor_indices[0])
                    else:
                        continue
                else:
                    continue
            if display: 
                cv2.imshow('results', out_img[:,:,::-1])
                cv2.waitKey(10)
            else:
                writer.append_data(out_img)
        
    if not display:
        writer.close()


np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])
def visualize_detection_results(img_np, active_actors, prob_dict):
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    #for ii in range(len(active_actors)):
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][-16], cur_actor['all_scores'][0], 1
        except IndexError:
            continue
        
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
        action_message_list = ["%s:%.3f" % (actres[0][0:7], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        # action_message = " ".join(action_message_list)

        raw_colors = COLORS[actor_id]
        rect_color = tuple(int(raw_color) for raw_color in raw_colors)
        text_color = tuple(255-color_value for color_value in rect_color)

        cv2.rectangle(disp_img, (left,top), (right,bottom), rect_color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), rect_color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, text_color, 1)

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), rect_color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, text_color, 1)

    return disp_img


def visualize_cams(image, input_frames, out_dict, actor_idx):
    #classes = ["walk", "bend", "carry"]
    #classes = ["sit", "ride"]
    classes = ["talk to", "watch (a", "listen to"]
    action_classes = [cc for cc in range(60) if any([cname in act.ACTION_STRINGS[cc] for cname in classes])]

    feature_activations = out_dict['final_i3d_feats']
    cls_weights = out_dict['cls_weights']
    input_frames = out_dict['cropped_frames'].astype(np.uint8)
    probs = out_dict["pred_probs"]

    class_maps = np.matmul(feature_activations, cls_weights)
    min_val = np.min(class_maps[:,:, :, :, :])
    max_val = np.max(class_maps[:,:, :, :, :]) - min_val

    normalized_cmaps = np.uint8((class_maps-min_val)/max_val * 255.)

    t_feats = feature_activations.shape[1]
    t_input = input_frames.shape[1]
    index_diff = (t_input) // (t_feats+1)

    img_new_height = 400
    img_new_width = int(image.shape[1] / float(image.shape[0]) * img_new_height)
    img_to_show = cv2.resize(image.copy(), (img_new_width,img_new_height))[:,:,::-1]
    #img_to_concat = np.zeros((400, 800, 3), np.uint8)
    img_to_concat = np.zeros((400, 400, 3), np.uint8)
    for cc in range(len(action_classes)):
        cur_cls_idx = action_classes[cc]
        act_str = act.ACTION_STRINGS[action_classes[cc]]
        message = "%s:%%%.2d" % (act_str[:20], 100*probs[actor_idx, cur_cls_idx])
        for tt in range(t_feats):
            cur_cam = normalized_cmaps[actor_idx, tt,:,:, cur_cls_idx]
            cur_frame = input_frames[actor_idx, (tt+1) * index_diff, :,:,::-1]

            resized_cam = cv2.resize(cur_cam, (100,100))
            colored_cam = cv2.applyColorMap(resized_cam, cv2.COLORMAP_JET)

            overlay = cv2.resize(cur_frame.copy(), (100,100))
            overlay = cv2.addWeighted(overlay, 0.5, colored_cam, 0.5, 0)

            img_to_concat[cc*125:cc*125+100, tt*100:(tt+1)*100, :] = overlay
        cv2.putText(img_to_concat, message, (20, 13+100+125*cc), 0, 0.5, (255,255,255), 1)

    final_image = np.concatenate([img_to_show, img_to_concat], axis=1)
    return final_image[:,:,::-1]






    


if __name__ == '__main__':
    main()
