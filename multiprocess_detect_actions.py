import numpy as np
import cv2
import imageio
#import tensorflow as tf
import json

import os
import sys
import argparse

import object_detection.object_detector as obj
import action_detection.action_detector as act

from multiprocessing import Process, Queue

import time
#SHOW_CAMS = True
SHOW_CAMS = False
# Object classes
CAM_CLASSES = ["read", "answer phone", "carry", "text on", "drink", "eat"]
# Person state classes
CAM_CLASSES = ["walk", "stand", "sit", "bend", "run", "talk"]

#USE_WEBCAM = True

ACTION_FREQ = 8
#OBJ_BATCH_SIZE = 16 # with ssd-mobilenet2
#OBJ_BATCH_SIZE = 4 # with ssd-mobilenet2
#OBJ_BATCH_SIZE = 1 # with NAS, otherwise memory exhausts
DELAY = 60 # ms, this limits the input around 16 fps. This makes sense as the action model was trained with similar fps videos.
#OBJ_GPU = "0"
#ACT_GPU = "2"
#ACT_GPU = "1" # if using nas and/or high res input use different GPUs for each process

T = 32 # Timesteps

# separate process definitions

# frame reader
def read_frames(reader, frame_q, use_webcam):
    if use_webcam:
        time.sleep(15)
        frame_cnt = 0
        while True:
            #if frame_cnt % 5 == 0:    
            #    ret, frame = reader.read()
            #    cur_img = frame[:,:,::-1]
            #    frame_q.put(cur_img)
            #else:
            #    ret, frame = reader.read()
            ret, frame = reader.read()
            cur_img = frame[:,:,::-1] # bgr to rgb from opencv reader
            frame_q.put(cur_img)
            if frame_q.qsize() > 100:
                time.sleep(1)
            else:
                time.sleep(DELAY/1000.)
            #print(cur_img.shape)
    else:
        #for cur_img in reader: # this is imageio reader, it uses rgb
        nframes = reader.get_length()
        for ii in range(nframes):
            while frame_q.qsize() > 500: # so that we dont use huge amounts of memory
                time.sleep(1)
            cur_img = reader.get_next_data()
            frame_q.put(cur_img)
            #shape = cur_img.shape
            #noisy_img = np.uint8(cur_img.astype(np.float) + np.random.randn(*shape) * 20)
            #frame_q.put(noisy_img)
            if ii % 100 == 0:
                print("%i / %i frames in queue" % (ii, nframes))
        print("All %i frames in queue" % (nframes))

# # object detector and tracker
# def run_obj_det_and_track(frame_q, detection_q, det_vis_q):
#     import tensorflow as tf # there is a bug. if you dont import tensorflow within the process you cant use the same gpus for both processes.
#     os.environ['CUDA_VISIBLE_DEVICES'] = OBJ_GPU
#     main_folder = "./"
#     ## Best
#     # obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
#     ## Good and Faster
#     #obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')


#     print("Loading object detection model at %s" % obj_detection_graph)


#     obj_detector = obj.Object_Detector(obj_detection_graph)
#     tracker = obj.Tracker()
#     while True:
#         cur_img = frame_q.get()
#         expanded_img = np.expand_dims(cur_img, axis=0)
#         detection_list = obj_detector.detect_objects_in_np(expanded_img)
#         detection_info = [info[0] for info in detection_list]
#         tracker.update_tracker(detection_info, cur_img)
#         rois_np, temporal_rois_np = tracker.generate_all_rois()
#         actors_snapshot = []
#         for cur_actor in tracker.active_actors:
#             act_id = cur_actor['actor_id']
#             act_box = cur_actor['all_boxes'][-1][:]
#             act_score = cur_actor['all_scores'][-1]
#             actors_snapshot.append({'actor_id':act_id, 'all_boxes':[act_box], 'all_scores':[act_score]})
#         #print(len(actors_snapshot))
#         #if actors_snapshot:
#         #    detection_q.put([cur_img, actors_snapshot, rois_np, temporal_rois_np])
#         #    det_vis_q.put([cur_img, actors_snapshot])
#         detection_q.put([cur_img, actors_snapshot, rois_np, temporal_rois_np])
#         det_vis_q.put([cur_img, actors_snapshot])

def run_obj_det_and_track_in_batches(frame_q, detection_q, det_vis_q, obj_batch_size, obj_gpu):
    import tensorflow as tf # there is a bug. if you dont import tensorflow within the process you cant use the same gpus for both processes.
    os.environ['CUDA_VISIBLE_DEVICES'] = obj_gpu
    main_folder = "./"

    obj_detection_graph = "./object_detection/weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
    #obj_detection_graph = "./object_detection/weights/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb"



    print("Loading object detection model at %s" % obj_detection_graph)


    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker(timesteps=T)
    while True:
        img_batch = []
        for _ in range(obj_batch_size): 
            cur_img = frame_q.get()
            img_batch.append(cur_img)
        #expanded_img = np.expand_dims(cur_img, axis=0)
        expanded_img = np.stack(img_batch, axis=0)
        start_time = time.time()
        detection_list = obj_detector.detect_objects_in_np(expanded_img)
        end_time = time.time()
        print("%.3f second per image" % ((end_time-start_time) / float(obj_batch_size)) )
        for ii in range(obj_batch_size):
            cur_img = img_batch[ii]
            detection_info = [info[ii] for info in detection_list]
            tracker.update_tracker(detection_info, cur_img)
            rois_np, temporal_rois_np = tracker.generate_all_rois()
            actors_snapshot = []
            for cur_actor in tracker.active_actors:
                act_id = cur_actor['actor_id']
                act_box = cur_actor['all_boxes'][-1][:]
                act_score = cur_actor['all_scores'][-1]
                actors_snapshot.append({'actor_id':act_id, 'all_boxes':[act_box], 'all_scores':[act_score]})
            #print(len(actors_snapshot))
            #if actors_snapshot:
            #    detection_q.put([cur_img, actors_snapshot, rois_np, temporal_rois_np])
            #    det_vis_q.put([cur_img, actors_snapshot])
            detection_q.put([cur_img, actors_snapshot, rois_np, temporal_rois_np])
            det_vis_q.put([cur_img, actors_snapshot])

# Action detector
def run_act_detector(shape, detection_q, actions_q, act_gpu):
    import tensorflow as tf # there is a bug. if you dont import tensorflow within the process you cant use the same gpus for both processes.
    os.environ['CUDA_VISIBLE_DEVICES'] = act_gpu
    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn', timesteps=T)
    #ckpt_name = 'model_ckpt_RGB_soft_attn-16'
    #ckpt_name = 'model_ckpt_soft_attn_ava-23'
    #ckpt_name = 'model_ckpt_soft_attn_pooled_ava-52'
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'
    main_folder = "./"
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)

    #input_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf([T,H,W,3])
    memory_size = act_detector.timesteps - ACTION_FREQ
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory(shape, memory_size)
    
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    

    act_detector.restore_model(ckpt_path)

    processed_frames_cnt = 0

    while True:
        images = []
        for _ in range(ACTION_FREQ):
            cur_img, active_actors, rois_np, temporal_rois_np = detection_q.get()
            images.append(cur_img)
            #print("action frame: %i" % len(images))
        
        if not active_actors:
            prob_dict = {}
            if SHOW_CAMS:
                prob_dict = {"cams": visualize_cams({})} 
        else:
            # use the last active actors and rois vectors
            no_actors = len(active_actors)

            cur_input_sequence = np.expand_dims(np.stack(images, axis=0), axis=0)

            if no_actors > 14:
                no_actors = 14
                rois_np = rois_np[:14]
                temporal_rois_np = temporal_rois_np[:14]
                active_actors = active_actors[:14]

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

            out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
            probs = out_dict['pred_probs']

            if not SHOW_CAMS:
                # associate probs with actor ids
                print_top_k = 5
                prob_dict = {}
                for bb in range(no_actors):
                    act_probs = probs[bb]
                    order = np.argsort(act_probs)[::-1]
                    cur_actor_id = active_actors[bb]['actor_id']
                    print("Person %i" % cur_actor_id)
                    cur_results = []
                    for pp in range(print_top_k):
                        print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                        cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                    prob_dict[cur_actor_id] = cur_results
            else:
                # prob_dict = out_dict
                prob_dict = {"cams": visualize_cams(out_dict)} # do it here so it doesnt slow down visualization process
            
        processed_frames_cnt += ACTION_FREQ # each turn we process this many frames
        
        if processed_frames_cnt >= act_detector.timesteps / 2:
            # we are doing this so we can skip the initialization period
            # first frame needs timesteps / 2 frames to be processed before visualizing
            actions_q.put(prob_dict)
        
        #print(prob_dict.keys())



# Visualization
def run_visualization(writer, det_vis_q, actions_q, display):
    frame_cnt = 0
    # prob_dict = actions_q.get() # skip the first one

    durations = []
    fps_message = "FPS: 0"
    while True:
        start_time = time.time()
        cur_img, active_actors = det_vis_q.get()
        #print(len(active_actors))
        if frame_cnt % ACTION_FREQ == 0:
            prob_dict = actions_q.get()

        if not SHOW_CAMS:
            out_img = visualize_detection_results(cur_img, active_actors, prob_dict)
        else:
            # out_img = visualize_cams(cur_img, prob_dict)
            img_to_concat = prob_dict["cams"] #if "cams" in prob_dict else np.zeros((400, 400, 3), np.uint8)
            image = cur_img
            img_new_height = 400
            img_new_width = int(image.shape[1] / float(image.shape[0]) * img_new_height)
            img_to_show = cv2.resize(image.copy(), (img_new_width,img_new_height))[:,:,::-1]
            out_img = np.array(np.concatenate([img_to_show, img_to_concat], axis=1)[:,:,::-1])
        
    
        if display: 
            cv2.putText(out_img, fps_message, (25, 25), 0, 1, (255,0,0), 1)
            cv2.imshow('results', out_img[:,:,::-1])
            cv2.waitKey(DELAY//2)
            #cv2.waitKey(1)
        #else:
        writer.append_data(out_img)
        frame_cnt += 1
        
        # FPS info
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
        if len(durations) > 32: del durations[0]
        if frame_cnt % 16 == 0 :
            print("avg time per frame: %.3f" % np.mean(durations))
            fps_message = "FPS: %i" % int(1 / np.mean(durations))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, required=False, default="", help="The path to the video and if it is not provided, webcam will be used.")
    parser.add_argument('-d', '--display', type=str, required=False, default="True",help="The display flag where the results will be visualized using OpenCV.")
    parser.add_argument('-b', '--obj_batch_size', type=int, required=False, default=16, help="Batch size for the object detector. Depending on the model used and gpu memory size, this should be changed.")
    parser.add_argument('-o', '--obj_gpu', type=str, required=False, default="0", help="Which GPU to use for object detector. Uses CUDA_VISIBLE_DEVICES environment var. Could be the same with action detector but in that case obj batch size should be reduced.")
    parser.add_argument('-a', '--act_gpu', type=str, required=False, default="0", help="Which GPU to use for action detector. Uses CUDA_VISIBLE_DEVICES environment var. Could be the same with object detector but in that case obj batch size should be reduced.")

    args = parser.parse_args()
    use_webcam = args.video_path == ""
    display = (args.display == "True" or args.display == "true")

    obj_batch_size = args.obj_batch_size
    obj_gpu = args.obj_gpu
    act_gpu = args.act_gpu
    
    #actor_to_display = 6 # for cams

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    
    #out_vid_path = "./output_videos/%s_output.mp4" % (basename if not SHOW_CAMS else basename+'_cams_actor_%.2d' % actor_to_display)
    out_vid_path = "./output_videos/%s_output.mp4" % basename 
    out_vid_path = out_vid_path if not use_webcam else './output_videos/webcam_output.mp4' 

    # video_path = "./tests/chase1Person1View3Point0.mp4"
    # out_vid_path = 'output.mp4'

    main_folder = './'
    

    if use_webcam:
        print("Using webcam")
        reader = cv2.VideoCapture(0)
        ## We can set the input shape from webcam, I use the default 640x480 to achieve real-time
        #reader.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = reader.read()
        if ret:
            H,W,C = frame.shape
        else:
            H = 480
            W = 640
        fps = 1000//DELAY
    else:
        print("Reading video file %s" % video_path)
        reader = imageio.get_reader(video_path, 'ffmpeg')
        fps = reader.get_meta_data()['fps'] #// fps_divider
        W, H = reader.get_meta_data()['size']
        #T = tracker.timesteps
    print("H: %i, W: %i" % (H, W))
    #T = 32
    
    # fps_divider = 1
    print('Running actions every %i frame' % ACTION_FREQ)

    writer = imageio.get_writer(out_vid_path, fps=fps)
    print("Writing output to %s" % out_vid_path)
    shape = [T,H,W,3]

    

    frame_q = Queue()
    detection_q = Queue()
    det_vis_q = Queue()
    actions_q = Queue()

    frame_reader_p = Process(target=read_frames, args=(reader, frame_q, use_webcam))
    #obj_detector_p = Process(target=run_obj_det_and_track, args=(frame_q, detection_q, det_vis_q))
    obj_detector_p = Process(target=run_obj_det_and_track_in_batches, args=(frame_q, detection_q, det_vis_q, obj_batch_size, obj_gpu))
    action_detector_p = Process(target=run_act_detector, args=(shape, detection_q, actions_q, act_gpu))
    visualization_p = Process(target=run_visualization, args=(writer, det_vis_q, actions_q, display))

    processes = [frame_reader_p, obj_detector_p, action_detector_p, visualization_p]

    for process in processes:
        process.daemon = True
        process.start()

    try:
        if use_webcam:
            while True:
                time.sleep(1)
                print("frame_q: %i, obj_q: %i, act_q: %i, vis_q: %i" % (frame_q.qsize(), detection_q.qsize(), actions_q.qsize(), det_vis_q.qsize()))
        else:
            time.sleep(5)
            while True:
                time.sleep(1)
                print("frame_q: %i, obj_q: %i, act_q: %i, vis_q: %i" % (frame_q.qsize(), detection_q.qsize(), actions_q.qsize(), det_vis_q.qsize()))
                if frame_q.qsize() == 0 and detection_q.qsize() == 0 and actions_q.qsize() == 0: # if all the queues are empty, we are done
                    writer.close()
                    break
    except KeyboardInterrupt:
        writer.close()
        if use_webcam:
            reader.release()
    print("Done!")



np.random.seed(10)
COLORS = np.random.randint(0, 100, [1000, 3]) # get darker colors for bboxes and use white text
def visualize_detection_results(img_np, active_actors, prob_dict):
    #score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    #for ii in range(len(active_actors)):
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        cur_box, cur_score, cur_class = cur_actor['all_boxes'][-1], cur_actor['all_scores'][-1], 1
        
        #if cur_score < score_th: 
        #    continue

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
        action_message_list = ["%s:%.3f" % (actres[0][:20], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        # action_message = " ".join(action_message_list)

        color = COLORS[actor_id]

        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        #cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255,255,255), 1)

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            #cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (255,255,255)-color, 1)
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (255,255,255), 1)

    return disp_img

#def visualize_cams(image, out_dict):#, actor_idx):
def visualize_cams(out_dict):#, actor_idx):
    # img_new_height = 400
    # img_new_width = int(image.shape[1] / float(image.shape[0]) * img_new_height)
    # img_to_show = cv2.resize(image.copy(), (img_new_width,img_new_height))[:,:,::-1]
    ##img_to_concat = np.zeros((400, 800, 3), np.uint8)
    #img_to_concat = np.zeros((400, 400, 3), np.uint8)
    if len(CAM_CLASSES) < 4:
        w = 400
    else:
        w = 900
    img_to_concat = np.zeros((400, w, 3), np.uint8)

    if out_dict:
        actor_idx = 0
        action_classes = [cc for cc in range(60) if any([cname in act.ACTION_STRINGS[cc] for cname in CAM_CLASSES])]

        feature_activations = out_dict['final_i3d_feats']
        cls_weights = out_dict['cls_weights']
        input_frames = out_dict['cropped_frames'].astype(np.uint8)
        probs = out_dict["pred_probs"]

        class_maps = np.matmul(feature_activations, cls_weights)
        #min_val = np.min(class_maps[:,:, :, :, :])
        #max_val = np.max(class_maps[:,:, :, :, :]) - min_val
        min_val = -200.
        max_val = 300.
        normalized_cmaps = (class_maps-min_val)/max_val * 255.
        normalized_cmaps[normalized_cmaps>255] = 255
        normalized_cmaps[normalized_cmaps<0] = 0
        normalized_cmaps = np.uint8(normalized_cmaps)

        #normalized_cmaps = np.uint8((class_maps-min_val)/max_val * 255.)

        t_feats = feature_activations.shape[1]
        t_input = input_frames.shape[1]
        index_diff = (t_input) // (t_feats+1)

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
                
                if cc > 2:
                    xx = tt + 5 # 4 timesteps visualized per class + 1 empty space
                    yy = cc - 3 # 3 classes per column
                else:
                    xx = tt
                    yy = cc

                img_to_concat[yy*125:yy*125+100, xx*100:(xx+1)*100, :] = overlay
            cv2.putText(img_to_concat, message, (20+int(cc>2)*500, 13+100+125*yy), 0, 0.5, (255,255,255), 1)

    return img_to_concat
    #final_image = np.concatenate([img_to_show, img_to_concat], axis=1)
    #return np.array(final_image[:,:,::-1])
    #return final_image


if __name__ == '__main__':
    main()
