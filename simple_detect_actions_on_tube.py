import cv2
import os
import numpy as np
import imageio

import action_detection.action_detector as act



def set_up_detector():

    act_detector = act.Action_Detector('soft_attn')
    #ckpt_name = 'model_ckpt_RGB_soft_attn-16'
    #ckpt_name = 'model_ckpt_soft_attn_ava-23'
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'


    input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()

    ckpt_path = os.path.join('action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)

    detector_dict = {   'detector':act_detector, 
                        'input_seq': input_seq,
                        'rois': rois,
                        'roi_batch_indices': roi_batch_indices,
                        'pred_probs': pred_probs}

    return detector_dict


def detect_on_tube(input_tube, detector_dict):
    """ Input tube has to be of shape batch_size x 32 x 400 x 400 x 3
        In this function I am assuming the actor is centered in the tube and the tube is a larger context
        ex: center 200x200 is the actor and remaining outside area is context"""
        
    batch_size = input_tube.shape[0]
    
    # assuming actors are centered 
    # (this should be changed depending on how much of context is included in the tube)

    rois_np = np.array([[0.25,0.25,0.75,0.75]]*batch_size)
    roi_batch_indices_np = np.arange(batch_size)

    act_detector = detector_dict['detector']
    # inputs
    input_seq_tf = detector_dict['input_seq']
    rois_tf = detector_dict['rois']
    roi_batch_indices_tf = detector_dict['roi_batch_indices']

    # output
    predictions_tf = detector_dict['pred_probs']

    feed_dict = {   input_seq_tf: input_tube,
                    rois_tf: rois_np,
                    roi_batch_indices_tf: roi_batch_indices_np}

    prediction_probabilites = act_detector.session.run(predictions_tf, feed_dict=feed_dict)

    return prediction_probabilites


def main():
    reader = imageio.get_reader("person_0_tube.mp4")
    frames = []
    for cur_frame in reader:
        frames.append(cur_frame)

    input_tube = np.stack(frames[:32], axis=0)
    input_tube = np.expand_dims(input_tube, axis=0) # batch dimension
    detector_dict = set_up_detector()
    prediction_probabilites = detect_on_tube(input_tube, detector_dict)

    top_k = 5
    top_classes = np.argsort(prediction_probabilites[0,:])[:-top_k-1:-1]

    print("Results")
    for ii in range(top_k):
        class_id = top_classes[ii]
        class_str = act.ACTION_STRINGS[class_id]
        class_prob = prediction_probabilites[0,class_id]
        print("%.10s : %.3f" % (class_str, class_prob))
    
    cv2.imshow('midframe', input_tube[0,16,:,:,::-1])
    cv2.waitKey(0)

    




if __name__ == '__main__':
    main()





    