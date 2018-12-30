import tensorflow as tf
import numpy as np

import action_detection.i3d as i3d

class Action_Detector():
    def __init__(self, model_arc, timesteps=32, session=None):
        self.architecture_str = model_arc
        

        self.is_training = tf.constant(False)
        self.num_classes = 60
        self.input_size = [400,400]
        self.timesteps = timesteps
        self.max_rois = 1
        self.act_graph = tf.Graph()

        if not session:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            
            session = tf.Session(graph=self.act_graph, config=config)
        self.session = session

    def restore_model(self, ckpt_file):
        self.session.run(self.init_op)
        print("Loading weights from %s" % ckpt_file)
        with self.act_graph.as_default():
            action_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ActionDetector')
            var_map = {}
            for variable in action_vars:
                map_name = variable.name.replace(':0', '')
                map_name = map_name.replace('ActionDetector/', '')
                var_map[map_name] = variable
            model_saver = tf.train.Saver(var_list=var_map)
        model_saver.restore(self.session, ckpt_file)
        

    def define_inference(self, input_seq, rois, roi_batch_indices):

        with tf.variable_scope('ActionDetector'):

            end_point = 'Mixed_4f'
            box_size = [10,10]

            i3d_model = i3d.I3D_model(modality='RGB', num_classes=self.num_classes)
            features, end_points = i3d_model.inference(input_seq, self.is_training, end_point)

            print('Using model %s' % self.architecture_str)
            if self.architecture_str == 'i3d_tail':
                box_features = temporal_roi_cropping(features, rois, roi_batch_indices, box_size)
                class_feats = self.i3d_tail_model(box_features)
            elif self.architecture_str == 'non_local_v1':
                box_features = temporal_roi_cropping(features, rois, roi_batch_indices, box_size)
                class_feats = self.non_local_ROI_model(box_features, features, roi_batch_indices)
            elif self.architecture_str == 'non_local_attn':
                box_features = temporal_roi_cropping(features, rois, roi_batch_indices, box_size)
                class_feats = self.non_local_ROI_feat_attention_model(box_features, features, roi_batch_indices)
            elif self.architecture_str == 'soft_attn':
                class_feats = self.soft_roi_attention_model(features, rois, roi_batch_indices, box_size)
            elif self.architecture_str == 'non_local_v2':
                class_feats = self.non_local_ROI_model_v2(features, rois, roi_batch_indices, box_size)
            else:
                print('Architecture not implemented!')
                raise NotImplementedError

            logits = tf.layers.dense(inputs=class_feats, 
                                    units=self.num_classes, 
                                    activation=None, 
                                    name='CLS_Logits', 
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

            pred_probs = tf.nn.sigmoid(logits)

            self.init_op = tf.global_variables_initializer()

            # tf.add_to_collection('debug', [features, box_features, class_feats, logits, pred_probs])

        return pred_probs

    def define_inference_with_placeholders(self):
        with self.act_graph.as_default():
            input_seq = tf.placeholder(tf.float32, [None, self.timesteps]+ self.input_size + [3])
            rois = tf.placeholder(tf.float32, [None, 4]) # top, left, bottom, right
            roi_batch_indices = tf.placeholder(tf.int32, [None])
            
            pred_probs = self.define_inference(input_seq, rois, roi_batch_indices)
        return input_seq, rois, roi_batch_indices, pred_probs
    
    def define_inference_with_placeholders_noinput(self, input_seq):
        with self.act_graph.as_default():
            rois = tf.placeholder(tf.float32, [None, 4]) # top, left, bottom, right
            roi_batch_indices = tf.placeholder(tf.int32, [None])
            
            pred_probs = self.define_inference(input_seq, rois, roi_batch_indices)
        return rois, roi_batch_indices, pred_probs
        

    ####### MODELS #######
    def basic_model(self, roi_box_features):
        # basic model, takes the input feature and averages across temporal dim
        # temporal_len = roi_box_features.shape[1]
        B, temporal_len, H, W, C = roi_box_features.shape
        avg_features = tf.nn.avg_pool3d(      roi_box_features,
                                              ksize=[1, temporal_len, 1, 1, 1],
                                              strides=[1, temporal_len, 1, 1, 1],
                                              padding='VALID',
                                              name='TemporalPooling')
        # classification
        class_feats = tf.layers.flatten(avg_features)

        return class_feats

    def basic_model_pooled(self, roi_box_features):
        # basic model, takes the input feature and averages across temporal dim
        # temporal_len = roi_box_features.shape[1]
        B, temporal_len, H, W, C = roi_box_features.shape
        avg_features = tf.nn.avg_pool3d(      roi_box_features,
                                              ksize=[1, temporal_len, H, W, 1],
                                              strides=[1, temporal_len, H, W, 1],
                                              padding='VALID',
                                              name='TemporalPooling')
        # classification
        class_feats = tf.layers.flatten(avg_features)

        return class_feats

    def i3d_tail_model(self, roi_box_features):
        # I3D continued after mixed4e
        with tf.variable_scope('Tail_I3D'):
            tail_end_point = 'Mixed_5c'
            # tail_end_point = 'Mixed_4f'
            final_i3d_feat, end_points = i3d.i3d_tail(roi_box_features, self.is_training, tail_end_point)
            # final_i3d_feat = end_points[tail_end_point]
            tf.add_to_collection('final_i3d_feats', final_i3d_feat)
            
            
            # flat_feats = self.spatio_temporal_averaging(final_i3d_feat)
            # flat_feats = tf.layers.flatten(final_i3d_feat)
            flat_feats = self.basic_model_pooled(final_i3d_feat)
            # import pdb;pdb.set_trace()
            pass
            

        return flat_feats
    
    def non_local_ROI_model(self, roi_box_features, context_features, cur_b_idx):
        '''
        roi_box_features: bounding box features extracted on detected people
        context_features: main feature map extracted from full frame
        cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
        '''
        with tf.variable_scope('Non_Local_Block'):
            _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
            R = tf.shape(roi_box_features)[0]
            _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
            B = tf.shape(context_features)[0]


            feature_map_channel = Cr / 4

            roi_embedding = tf.layers.conv3d(roi_box_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
            context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

            context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')

            # Number of rois(R) is larger than number of batches B as from each segment we extract multiple rois
            # we need to gather batches such that rois are assigned to correct context features that they were extracted from
            context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')
            context_response_gathered = tf.gather(context_response, cur_b_idx, axis=0, name='ContextResGather')
            # now they have R as first dimensions

            # reshape so that we can use matrix multiplication to calculate attention mapping
            roi_emb_reshaped = tf.reshape(roi_embedding, shape=[R, Tr*Hr*Wr, feature_map_channel])
            context_emb_reshaped = tf.reshape(context_embedding_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])
            context_res_reshaped = tf.reshape(context_response_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])

            emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]
            emb_mtx = emb_mtx / tf.sqrt(tf.cast(feature_map_channel, tf.float32)) # normalization of rand variables

            embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

            attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

            attention_response_org_shape = tf.reshape(attention_response, [R, Tr, Hr, Wr, feature_map_channel])

            # blow it up to original feature dimension
            non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cr, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

            # Residual connection
            residual_feature = roi_box_features + non_local_feature

        i3d_tail_feats = self.i3d_tail_model(residual_feature)

        return i3d_tail_feats

    def non_local_ROI_feat_attention_model(self, roi_box_features, context_features, cur_b_idx):
        '''
        roi_box_features: bounding box features extracted on detected people
        context_features: main feature map extracted from full frame
        cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
        '''
        with tf.variable_scope('Non_Local_Block'):
            _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
            R = tf.shape(roi_box_features)[0]
            _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
            B = tf.shape(context_features)[0]


            feature_map_channel = Cr / 64

            roi_embedding = tf.layers.conv3d(roi_box_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
            # roi_embedding = roi_box_features

            context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

            context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')

            # Number of rois(R) is larger than number of batches B as from each segment we extract multiple rois
            # we need to gather batches such that rois are assigned to correct context features that they were extracted from
            context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')
            context_response_gathered = tf.gather(context_response, cur_b_idx, axis=0, name='ContextResGather')
            # now they have R as first dimensions

            # reshape so that we can use matrix multiplication to calculate attention mapping
            roi_emb_reshaped = tf.reshape(roi_embedding, shape=[R, Tr*Hr*Wr, feature_map_channel, 1])
            roi_emb_permuted = tf.transpose(roi_emb_reshaped, perm=[0,2,1,3]) # [R, feature_map_channel, Tr*Hr*Wr, 1]
            
            context_emb_reshaped = tf.reshape(context_embedding_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel, 1])
            context_emb_permuted = tf.transpose(context_emb_reshaped, perm=[0,2,1,3]) # [R, feature_map_channel, Tc*Hc*Wc, 1]

            context_res_reshaped = tf.reshape(context_response_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])
            context_res_permuted = tf.transpose(context_res_reshaped, perm=[0,2,1]) # [R, feature_map_channel, Tc*Hc*Wc]

            emb_mtx = tf.matmul(roi_emb_permuted, context_emb_permuted, transpose_b=True) # [R, feature_map_channel,Tr*Hr*Wr, Tc*Hc*Wc]
            # emb_mtx = emb_mtx / tf.sqrt(tf.cast(feature_map_channel, tf.float32)) # normalization of rand variables

            embedding_mtx_permuted = tf.transpose(emb_mtx, [0, 2, 1, 3]) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

            embedding_attention = tf.nn.softmax(embedding_mtx_permuted, name='EmbeddingNormalization') # get the weights

            context_res_expanded = tf.expand_dims(context_res_permuted, axis=1)
            context_res_tiled = tf.tile(context_res_expanded, [1,Tr*Hr*Wr,1,1]) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

            attention_response = tf.multiply(embedding_attention, context_res_tiled) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

            attention_response_reduced = tf.reduce_sum(attention_response, axis=3) # this final sum gives the weighted sum of context_responses

            attention_response_org_shape = tf.reshape(attention_response_reduced, [R, Tr, Hr, Wr, feature_map_channel])

            # blow it up to original feature dimension
            non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cr, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

            # Residual connection
            residual_feature = roi_box_features + non_local_feature

        i3d_tail_feats = self.i3d_tail_model(residual_feature)

        return i3d_tail_feats

    def soft_roi_attention_model(self, context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE):
        with tf.variable_scope('Soft_Attention_Model'):
            _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
            B = tf.shape(context_features)[0]
            feature_map_channel = Cc / 4

            roi_box_features = temporal_roi_cropping(context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
            R = tf.shape(shifted_rois)[0]

            flat_box_feats = self.basic_model(roi_box_features)
            roi_embedding = tf.layers.dense(flat_box_feats, 
                                            feature_map_channel, 
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                            name='RoiEmbedding')

            context_embedding = tf.layers.conv3d(context_features, 
                                                filters=feature_map_channel, 
                                                kernel_size=[1,1,1], 
                                                padding='SAME', 
                                                activation=tf.nn.relu, 
                                                name='ContextEmbedding')
            
            with tf.device('/cpu:0'):
                roi_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(roi_embedding, axis=1), axis=1), axis=1) # R,512 -> R,1,1,1,512
                roi_tiled = tf.tile(roi_expanded, [1,Tc,Hc,Wc,1], 'RoiTiling')

                # multiply context_feats by no of rois so we can concatenate
                context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')

                roi_context_feats = tf.concat([roi_tiled, context_embedding_gathered], 4, name='RoiContextConcat')

            relation_feats = tf.layers.conv3d(  roi_context_feats, 
                                                filters=Cc, 
                                                kernel_size=[1,1,1], 
                                                padding='SAME', 
                                                activation=None, 
                                                name='RelationFeats')
            
            attention_map = tf.nn.sigmoid(relation_feats,'AttentionMap') # use sigmoid so it represents a heatmap of attention
            
            # with tf.device('/cpu:0'):
            # Multiply attention map with context features. Now this new feature represents the roi
            gathered_context = tf.gather(context_features, cur_b_idx, axis=0, name='ContextGather')
            soft_attention_feats = tf.multiply(attention_map, gathered_context)
        class_feats = self.i3d_tail_model(soft_attention_feats)
        return class_feats

        # with tf.variable_scope('Tail_I3D'):
        #     tail_end_point = 'Mixed_5c'
        #     # tail_end_point = 'Mixed_4f'
        #     final_i3d_feat, end_points = i3d.i3d_tail(soft_attention_feats, self.is_training, tail_end_point)
        
        # temporal_len = final_i3d_feat.shape[1]
        # avg_features = tf.nn.avg_pool3d(      final_i3d_feat,
        #                                       ksize=[1, temporal_len, 3, 3, 1],
        #                                       strides=[1, temporal_len, 3, 3, 1],
        #                                       padding='SAME',
        #                                       name='TemporalPooling')
        # # classification
        # class_feats = tf.layers.flatten(avg_features)

        # return class_feats

    def non_local_ROI_model_v2(self, context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE):
        '''
        roi_box_features: bounding box features extracted on detected people
        context_features: main feature map extracted from full frame
        cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
        '''
        with tf.variable_scope('Non_Local_Block'):
            # _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
            # R = tf.shape(roi_box_features)[0]
            _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
            B = tf.shape(context_features)[0]

            feature_map_channel = Cc / 2

            roi_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
            context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

            context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')


            # reshape so that we can use matrix multiplication to calculate attention mapping
            roi_emb_reshaped = tf.reshape(roi_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
            context_emb_reshaped = tf.reshape(context_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
            context_res_reshaped = tf.reshape(context_response, shape=[B, Tc*Hc*Wc, feature_map_channel])

            emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]

            embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

            attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

            attention_response_org_shape = tf.reshape(attention_response, [B, Tc, Hc, Wc, feature_map_channel])

            # blow it up to original feature dimension
            non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cc, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

            # Residual connection
            residual_feature = context_features + non_local_feature

        box_features = temporal_roi_cropping(residual_feature, shifted_rois, cur_b_idx, BOX_CROP_SIZE)

        i3d_tail_feats = self.i3d_tail_model(box_features)

        return i3d_tail_feats


    def crop_tubes_in_tf(self, frame_shape):
        T,H,W,C = frame_shape
        with self.act_graph.as_default():
            input_frames = tf.placeholder(tf.float32, [None, T, H,W,C])
            rois = tf.placeholder(tf.float32, [None, T, 4]) # top, left, bottom, right
            batch_indices = tf.placeholder(tf.int32, [None])

            # use temporal rois since we track the actor over time
            cropped_frames = temporal_roi_cropping(input_frames, rois, batch_indices, self.input_size, temp_rois=True)
    
        return input_frames, rois, batch_indices, cropped_frames
    
    def crop_tubes_in_tf_with_memory(self, frame_shape, memory_size):
        T,H,W,C = frame_shape
        no_updated_frames = T - memory_size
        with self.act_graph.as_default():
            updated_frames, combined_sequence = memory_placeholder(tf.float32, [1, T, H,W,C], memory_size)
            rois = tf.placeholder(tf.float32, [None, T, 4]) # top, left, bottom, right
            batch_indices = tf.placeholder(tf.int32, [None])

            # use temporal rois since we track the actor over time
            cropped_frames = temporal_roi_cropping(combined_sequence, rois, batch_indices, self.input_size, temp_rois=True)
    
        return updated_frames, rois, batch_indices, cropped_frames

def memory_placeholder(dtype, shape, memory_size, name=None):
    ''' Every time we run the action detector we need to upload all 32 frames to the GPUs with feeddict
        This is slow and redundant as most of the frames are shared with the previous run.
        Why not just shift the frames?
        about 30-40% speed improvement
    '''
    T = shape[1]

    no_updated_frames = T - memory_size
    updated_shape = [1, no_updated_frames]+shape[2:]
    updated_frames = tf.placeholder(dtype, updated_shape)

    memory_shape = [1, memory_size] + shape[2:]
    memory_frames = tf.Variable(initial_value=np.zeros(memory_shape), dtype=np.float32, trainable=False)

    combined_sequence = tf.concat([memory_frames, updated_frames], axis=1)
    # update operation for the memory, move the new frames to the memory for the next run
    update_memory = tf.assign(memory_frames, combined_sequence[0:1, -memory_size:])

    with tf.get_default_graph().control_dependencies([update_memory]):
        combined_sequence = tf.identity(combined_sequence)

    return updated_frames, combined_sequence





def temporal_roi_cropping(features, rois, batch_indices, crop_size, temp_rois=False):
# def temporal_roi_cropping(features, rois, batch_indices, crop_size):
    ''' features is of shape [Batch, T, H, W, C]
        rois [num_boxes, TEMP_RESOLUTION, 4] or [num_boxes, 4] depending on temp_rois flag
        batch_indices [num_boxes]
    '''
    # import pdb;pdb.set_trace()
    B = tf.shape(features)[0]
    _, T, H, W, C = features.shape.as_list()
    num_boxes = tf.shape(rois)[0]
 
    if temp_rois:
        # slope = (T-1) / tf.cast(TEMP_RESOLUTION - 1, tf.float32)
        # indices = tf.cast(slope * tf.range(TEMP_RESOLUTION, dtype=tf.float32), tf.int32)
        # slope = (TEMP_RESOLUTION-1) / float(T - 1)
        # indices = (slope * np.arange(T)).astype(np.int32)
        # temporal_rois = tf.gather(rois,indices,axis=1,name='temporalsampling')
        temporal_rois = rois
    else:
        # use the keyframe roi for all time indices
        temporal_rois = tf.expand_dims(rois, axis=1)
        temporal_rois = tf.tile(temporal_rois, [1, T, 1])
    # temporal_rois = tf.expand_dims(rois, axis=1)
    # temporal_rois = tf.tile(temporal_rois, [1, T, 1])
 
    # since we are flattening the temporal dimension and batch dimension
    # into a single dimension, we need new batch_index mapping
     
    # batch_indices = [0,0,1,1,2]
    temporal_mapping = batch_indices * T # gives the starting point for each sample in batch
    # temporal_mapping = [0,0,16,16,32]
     
    temporal_mapping = tf.expand_dims(temporal_mapping, axis=1)
    # temporal_mapping = [0,0,16,16,32]
     
    temporal_mapping = tf.tile(temporal_mapping, [1, T])
    #   [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]],
     
    temporal_mapping = temporal_mapping + tf.range(T)
    #    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    #    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
 
    temporal_mapping = tf.reshape(temporal_mapping, [-1])
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,
    #     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18,
    #    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19,
    #    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    #    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
 
 
    # combine temporal dimension with batch dimension
    #stacked_features = tf.transpose(features, perm=[1, 0, 2, 3, 4])
    stacked_features = tf.reshape(features, [-1, H, W, C])
 
    # combine rois and mappings
    #stacked_rois = tf.transpose(rois, perm=[1,0,2])
    stacked_rois = tf.reshape(temporal_rois, [-1, 4])
 
    #stacked_mapping = tf.transpose(mapping, perm=[1,0])
    stacked_mapping = tf.reshape(temporal_mapping, [-1])
 
    ## cropped boxes 
    cropped_boxes = tf.image.crop_and_resize(image=stacked_features, 
                                             boxes=stacked_rois,
                                             box_ind=stacked_mapping,
                                             crop_size=crop_size
                                             )

    # ## Bilinearly crop first and then take max pool. This in theory would work better with sparse feats
    # double_size = [cc*2 for cc in crop_size]
    # bilinear_cropped_boxes = tf.image.crop_and_resize(  image=stacked_features, 
    #                                                     boxes=stacked_rois,
    #                                                     box_ind=stacked_mapping,
    #                                                     crop_size=double_size
    #                                                     )
    # cropped_boxes = tf.layers.max_pooling2d(bilinear_cropped_boxes, [2,2], [2,2], padding='valid')

    # now it has shape B*T, crop size
    # cropped_boxes = tf.Print(cropped_boxes, [tf.shape(cropped_boxes)], 'cropped shape')
    # unrolled_boxes = tf.reshape(cropped_boxes, [T, num_boxes, crop_size[0], crop_size[1], C])
    unrolled_boxes = tf.reshape(cropped_boxes, [num_boxes, T, crop_size[0], crop_size[1], C])
 
    # swap the boxes and time dimension
    # boxes = tf.transpose(unrolled_boxes, perm=[1, 0, 2, 3, 4])
    boxes = unrolled_boxes
 
    return boxes #, stacked_features

ACTION_STRINGS = { \
0: "bend/bow (at the waist)",
1: "crouch/kneel",
2: "dance",
3: "fall down",
4: "get up",
5: "jump/leap",
6: "lie/sleep",
7: "martial art",
8: "run/jog",
9: "sit",
10: "stand",
11: "swim",
12: "walk",
13: "answer phone",
14: "carry/hold (an object)",
15: "climb (e.g., a mountain)",
16: "close (e.g., a door, a box)",
17: "cut",
18: "dress/put on clothing",
19: "drink",
20: "drive (e.g., a car, a truck)",
21: "eat",
22: "enter",
23: "hit (an object)",
24: "lift/pick up",
25: "listen (e.g., to music)",
26: "open (e.g., a window, a car door)",
27: "play musical instrument",
28: "point to (an object)",
29: "pull (an object)",
30: "push (an object)",
31: "put down",
32: "read",
33: "ride (e.g., a bike, a car, a horse)",
34: "sail boat",
35: "shoot",
36: "smoke",
37: "take a photo",
38: "text on/look at a cellphone",
39: "throw",
40: "touch (an object)",
41: "turn (e.g., a screwdriver)",
42: "watch (e.g., TV)",
43: "work on a computer",
44: "write",
45: "fight/hit (a person)",
46: "give/serve (an object) to (a person)",
47: "grab (a person)",
48: "hand clap",
49: "hand shake",
50: "hand wave",
51: "hug (a person)",
52: "kiss (a person)",
53: "lift (a person)",
54: "listen to (a person)",
55: "push (another person)",
56: "sing to (e.g., self, a person, a group)",
57: "take (an object) from (a person)",
58: "talk to (e.g., self, a person, a group)",
59: "watch (a person)" }
