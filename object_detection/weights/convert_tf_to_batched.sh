# make sure object_detection api is built using: protoc object_detection/protos/*.proto --python_out=.
# .config files are located in models/object_detection/samples/configs/
#MODEL_NAME=faster_rcnn_nas_lowproposals_coco_2018_01_28
#MODEL_NAME=faster_rcnn_resnet101_ava_v2.1_2018_04_30
#MODEL_NAME=faster_rcnn_nas_coco_2018_01_28
MODEL_NAME=faster_rcnn_resnet50_coco_2018_01_28
MODEL_PATH=$PWD/tf_zoo/$MODEL_NAME
OUT_PATH=$PWD/batched_zoo/$MODEL_NAME
python ../models/research/object_detection/export_inference_graph.py --input_type image_tensor \
							   --pipeline_config_path $MODEL_PATH/pipeline.config \
							   --trained_checkpoint_prefix $MODEL_PATH/model.ckpt \
							   --output_directory $OUT_PATH/batched_graph
