import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import action_detection.action_detector as act

def test_on_local_segment():
    actors = [0,1,2]
    size = [400,400]
    timesteps = 32
    batch_np = np.zeros([len(actors), timesteps] + size + [3])
    rois_np = np.zeros([len(actors), 4])
    batch_indices_np = np.array(range(len(actors)))

    for bb, actor_id in enumerate(actors):
        vid_path = 'person_%i.mp4' % actor_id
        reader = imageio.get_reader(vid_path, 'ffmpeg')
        for tt, frame in enumerate(reader):
            batch_np[bb,tt,:] = frame
        
        roi_path = "person_%i_roi.json" % actor_id
        with open(roi_path) as fp:
            rois_np[bb] = json.load(fp)
    
    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn')
    ckpt_name = 'model_ckpt_RGB_soft_attn-9'
    input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()
    sess = act_detector.session

    #main_folder = sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main_folder = "../"
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)


    feed_dict = {input_seq:batch_np, rois:rois_np, roi_batch_indices:batch_indices_np}
    probs = sess.run(pred_probs, feed_dict=feed_dict)
    # debug = sess.run(tf.get_collection('debug'), feed_dict=feed_dict)
    # import pdb;pdb.set_trace()

    # highest_conf_actions = np.argsort(probs, axis=1)
    print_top_k = 5
    for ii in range(len(actors)):
        act_probs = probs[ii]
        order = np.argsort(act_probs)[::-1]
        print("Person %i" % actors[ii])
        for pp in range(print_top_k):
            print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))



if __name__ == '__main__':
    test_on_local_segment()
