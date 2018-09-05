import numpy as np
import cv2
import tensorflow as tf

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def test_local_image():

    test_img_path = 'chase.png'
    test_img = cv2.imread(test_img_path)
    detection_list, _, _ = run_on_image(test_img, None, None)
    out_img = visualize_results(test_img, detection_list, display=False)
    #import pdb;pdb.set_trace()
    cv2.imwrite('chase_out.jpg', out_img)
    obj_detection_graph = '/dccstor/srallap1/oytun/work/tensorflow_object/zoo/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb'


np.random.seed(10)
COLORS = np.random.randint(0, 255, [300, 3])
def visualize_results(img_np, detection_list, display=True):
    import cv2
    score_th = 0.30

    boxes,scores,classes,num_detections = [batched_term[0] for batched_term in detection_list]

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    for ii in range(num_detections):
        cur_box, cur_score, cur_class = boxes[ii], scores[ii], classes[ii]
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box


        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        #label = bbox['class_str']
        label = 'Class_%i' % cur_class
        message = label + '%% %.2f' % conf

        color = COLORS[ii]


        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)

    if display: 
        cv2.imshow('results', disp_img)
        cv2.waitKey(0)
    return disp_img
