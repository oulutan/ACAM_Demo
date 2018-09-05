import tensorflow as tf
import numpy as np

class Object_Detector():
    def __init__(self, graph_path, session=None):
        self.graph_path = graph_path
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.detection_graph = detection_graph

        if not session:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            
            session = tf.Session(graph=detection_graph, config=config)
        self.session = session

    def detect_objects_in_np(self, image_np):
        '''
        Runs the object detection on a single image or a batch of images.
        image_np can be a batch or a single image with batch dimension 1, dims:[None, None, None, 3]
        Returned boxes are top, left, bottom, right = current_bbox
        '''
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.session.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np[:,:,:,:]})
    
        return boxes,scores,classes,num_detections

    def detect_objects_in_tf(self):
        '''
        Returns the tensor pointers for the object detection inputs and outputs
        image_tensor can be a batch or a single image with batch dimension 1, dims:[None, None, None, 3]
        Returned boxes are top, left, bottom, right = current_bbox
        '''
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        return image_tensor, boxes, scores, classes, num_detections




