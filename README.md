# Conditional_Attention_Maps_Demo

This repo contains the demo code for our action recognition model explained in ARXIV LINK. 

This repo only contains the demo code, training and evaluation codes will be released in https://github.com/oulutan/ActorConditionedAttentionMaps .

The demo code achieves 16 fps through webcam using a GTX 1080Ti using multiprocessing and some other tricks. See the videos at YOUTUBE LINK. 

Additionally, we implemented the activation map displaying functionality to the demo code. This shows where the model gets activated for each action and gives pretty interesting results on understanding what model sees. 

The demo code includes a complete pipeline including object detection (Using TF Object API: https://github.com/tensorflow/models), tracking/bbox matching (Using DeepSort: https://github.com/nwojke/deep_sort) and our action detection module.

# Requirements
1. Tensorflow (Tested with 1.7 and 1.12)
2. Sonnet (DeepMind) for I3D backbone ``` pip install dm-sonnet ```
3. DeepSort (explained in next section)
4. TF object detection API (explained in next section)
5. OpenCV for webcam input and displaying purposes
6. ImageIO for video IO ```pip install imageio```. I rather use ImageIO for video IO than OpenCV as it is (a lot) easier to setup. This is being used when the input is a video file instead of webcam.

# How to run

1. Clone the repo recursively

```bash
git clone --recursive https://github.com/oulutan/ACAM_Demo
```

2. Compile Tensorflow Object Detection API within object_detection/models following https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

3. Make sure the object detection requirements are within PYTHONPATH. I have an easy script for this. 
```bash
# within /object_detection
source SET_ENVIRONMENT.sh
```

