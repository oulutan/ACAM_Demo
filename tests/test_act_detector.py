import numpy as np
import cv2
import imageio
import tensorflow as tf

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import action_detection.action_detector as act

def test_on_local_segment():
    vid_path = 'person_0.mp4'
    reader = imageio.get_reader(vid_path, 'ffmpeg')

    roi = []