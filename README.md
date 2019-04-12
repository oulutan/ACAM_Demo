# Actor Conditioned Attention Maps - Demo Repository

This repo contains the demo code for our action recognition model explained in https://arxiv.org/abs/1812.11631 

If you use this work, please cite our paper: 

```
@article{ulutan2018actor,
  title={Actor Conditioned Attention Maps for Video Action Detection},
  author={Ulutan, Oytun and Rallapalli, Swati and Torres,Carlos and Srivatsa, Mudhakar and Manjunath, BS},
  journal={arXiv preprint arXiv:1812.11631},
  year={2018}
}
```

Updated version of our paper is out! Check it out on Arxiv.

This repo only contains the demo code, training and evaluation codes will be released in https://github.com/oulutan/ActorConditionedAttentionMaps (Currently private repo)

The demo code achieves 16 fps through webcam using a GTX 1080Ti using multiprocessing and some other tricks. See the video at [real-time demo link](https://drive.google.com/open?id=1T5AJYp1cF0wLnxG8FmRjoUEGtiR7vvYh). Following is a snapshot from the video. 

<img src="https://github.com/oulutan/ACAM_Demo/blob/master/github_images/lab_actions_snap.png" width="500">


Additionally, we implemented the activation map displaying functionality to the demo code. This shows where the model gets activated for each action and gives pretty interesting results on understanding what model sees.

[Object interactions activation maps](https://drive.google.com/open?id=1Ly97R6HvFQMkZy9emvLlXTRN125HO2-R) 
![Object Snaphot](https://github.com/oulutan/ACAM_Demo/blob/master/github_images/object_cams_snap.png)


[Person movement actions activation maps](https://drive.google.com/open?id=1U2E1WvYlvKGmlbnVlOu8CWYhsApKygCR)
![Person Snaphot](https://github.com/oulutan/ACAM_Demo/blob/master/github_images/person_states_snap.png)

The demo code includes a complete pipeline including object detection (Using [TF Object API](https://github.com/tensorflow/models)), tracking/bbox matching (Using [DeepSort](https://github.com/nwojke/deep_sort)) and our action detection module.

# Requirements
1. Tensorflow (Tested with 1.7, 1.11, 1.12)
2. Sonnet 1.21 (DeepMind) for I3D backbone ``` pip install dm-sonnet==1.21 ```
3. DeepSort (explained in next section), you need to use my fork of it. Just clone this repo with --recursive
4. TF object detection API (explained in next section)
5. OpenCV (3.3.0) with contrib for webcam input and displaying purposes
6. ImageIO (2.4.1) for video IO ```pip install imageio==2.4.1```. I rather use ImageIO for video IO than OpenCV as it is (a lot) easier to setup for video support. This is being used when the input is a video file instead of webcam.

# Installation

1. Clone the repo recursively

```bash
git clone --recursive https://github.com/oulutan/ACAM_Demo
```

2. Compile Tensorflow Object Detection API within object_detection/models following https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

It should be just the protoc compilation like the following: 
```bash
# From object_detection/models/research/
protoc object_detection/protos/*.proto --python_out=.
```
If you are getting errors you have to download the required protoc and run that
```bash
# From object_detection/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```


3. Make sure the object detection API and Deepsort are within PYTHONPATH. I have an easy script for this. 
```bash
# From object_detection/
source SET_ENVIRONMENT.sh
```

4. Download and extract Tensorflow Object Detection models into object_detection/weights/ from: 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

16 Fps from webcam was achieved by 
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

5. Download DeepSort re-id model into object_detection/deep_sort/weights/ from their author's drive: 
https://drive.google.com/open?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo

6. Download the action detector model weights into action_detection/weights/ from following link:
https://drive.google.com/open?id=138gfVxWs_8LhHiVO03tKpmYBzIaTgD70

# How to run
There are 2 main scripts in this repo. 

```detect_actions.py``` is the simple and slow version where each module works sequentially and it is easier to understand. 

```multiprocess_detect_actions.py``` is the fast version where each module runs separately on their own process.

Arguments:
* --v (--video_path): The path to the video and if it is not provided, webcam will be used.
* -b (--obj_batch_size): Batch size for the object detector. Depending on the model used and gpu memory size, this should be changed. 
* -o (--obj_gpu): Which GPU to use for object detector. Uses "CUDA_VISIBLE_DEVICES" environment var. Could be the same with action detector but in that case obj batch size should be reduced. 
* -a (--act_gpu): Which GPU to use for action detector. Uses "CUDA_VISIBLE_DEVICES" environment var. Could be the same with object detector but in that case obj batch size should be reduced. 
* -d (--display): The display flag where the results will be visualized using OpenCV.

Object detection model can be replaced by any model in the API model zoo. Additionally, there is a object detection batch size parameter which should be changed depending on the GPU memory size and object model detection requirements. 

Object detection and Action detection can use different GPUs for faster performance. 

A sample input is included for testing purposes. Run the model on it using:
```
python multiprocess_detect_actions.py --video_path sample_input.mp4
```

UPDATE: We included a simple script to use the action detection directly. 

```simple_detect_actions_on_tube.py```

This script removes all the object detector/tracker parts from the code and only uses the action detector part. This should be your starting point if you want to switch out the object detectors or build something just using the action detector. 

This script only takes a tube as input. In this tube, it is assumed that person is centered and a larger context is also available. Context/actor ratio is defined in ```rois_np``` variable. (Currently assumes actor is in coordinates [0.25 - 0.75]).

A sample tube is provided. ```person_0_tube.mp4```

# How is real-time performance achieved?

1. Multiprocessing. Each module in the pipeline (Video input, Object detection/tracking, Action Detection and Video output) runs separately on different processes. Additional performance can be achieved by using separate gpus. 

2. Object Detector batched processing. Instead of running the object detector on each frame separately, performance was improved by batching multiple frames together. 

3. Input to the action detector is 32 frames and we have %75 overlap between their intervals. This means that every time we run action detectors, we only see 8 new frames. Uploading 32 frames to the gpu is a slow process without pre-fetching. Because of that, in this model, we keep the remaining 24 frames on the GPU as a tf.variable while updating the new 8, like a queue.

4. The planned use for this model was for camera views such as surveillance videos. Because of that, for each detected person, we crop a larger area centered on that person (last section on the paper) instead of using the whole input frame. However, uploading a different cropped region to the GPU for each detected person is also a slow process. So instead we upload the whole input and person locations which are cropped within tensorflow. 

5. SSD-MobileNetV2 is a fast detector. Additionally, the input webcam frame is limited to 640x480 resolution.
