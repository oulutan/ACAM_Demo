#use it with source SET_ENVIRONMENT.sh
#OBJ_PATH=/home/oytun/Dropbox/Python/tensorflow_object
OBJ_PATH=$PWD
TF_MODELS_PATH=$OBJ_PATH/models/research
export PYTHONPATH=$PYTHONPATH:$OBJ_PATH:$TF_MODELS_PATH:$TF_MODELS_PATH/slim:$TF_MODELS_PATH/object_detection:$OBJ_PATH/deep_sort
echo $PYTHONPATH

#source /home/oytun/python/envs/latestTF/bin/activate
