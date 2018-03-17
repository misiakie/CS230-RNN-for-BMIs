"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np
def input_fn(mode, filenames, params):
    """Input function for our model

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        filenames: (string) tfRecords files
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
        
    delay_time_min          = params.delay_time_min
    n_features_spikeRaster  = params.n_features_spikeRaster
    n_features_spikeRaster2 = params.n_features_spikeRaster2
    max_len_sequence        = params.max_len_sequence
    
    
    def _parse_function(example_proto):
        """
        Parses an example from a tfrecords file
        """
        features = {
                    "targetPos" : tf.FixedLenFeature([3],tf.float32),
                    'numTarget' : tf.FixedLenFeature([], tf.int64),
                    'spikeRaster': tf.SparseFeature(index_key=['spikeRaster_x', 'spikeRaster_y'],
                                                      value_key='spikeRaster_values',
                                                      dtype=tf.float32, size=[n_features_spikeRaster, max_len_sequence]),
                    'spikeRaster2': tf.SparseFeature(index_key=['spikeRaster2_x', 'spikeRaster2_y'],
                                                    value_key='spikeRaster2_values',
                                                    dtype=tf.float32, size=[n_features_spikeRaster, max_len_sequence]),
                    "spikeRaster_shape": tf.FixedLenFeature([2],tf.int64),
                    "isSuccessful": tf.FixedLenFeature([1], tf.int64),
                    "delayTime": tf.FixedLenFeature([1], tf.float32),
                    'timeTargetOn': tf.FixedLenFeature([1], tf.float32),                
                   }
        
        parsed_features = tf.parse_single_example(example_proto, features)
        
        # Predictive period => from timeTargetOn to delay_time_in
        begin_time   = tf.cast(parsed_features["timeTargetOn"], tf.int64)
        begin_sparse = tf.pad(begin_time, [[1,0]], 'CONSTANT')
        
        # Preprocess spikeRaster => [Time Series n_steps x n_features_spikeRaster]
        spikeRaster = tf.sparse_slice(parsed_features["spikeRaster"],
                                      begin_sparse,[n_features_spikeRaster,delay_time_min])
        spikeRaster = tf.sparse_tensor_to_dense(spikeRaster)
        spikeRaster = tf.transpose(spikeRaster)
        spikeRaster.set_shape((delay_time_min, n_features_spikeRaster))
    
        # Preprocess spikeRaster2 => [Time Series n_steps x n_features_spikeRaster]
        spikeRaster2 = tf.sparse_slice(parsed_features["spikeRaster2"],
                                       begin_sparse,[n_features_spikeRaster2,delay_time_min])
        spikeRaster2 = tf.sparse_tensor_to_dense(spikeRaster2)
        spikeRaster2 = tf.transpose(spikeRaster2)
        spikeRaster2.set_shape((delay_time_min, n_features_spikeRaster2))
    
        # Combine spikeRaster + spikeRaster2
        spikeRasters = tf.concat([spikeRaster,spikeRaster2], axis=1)
    
        # isSuccessful into a boolean
        isSuccessful = tf.cast(parsed_features["isSuccessful"], tf.bool)
        
        # target Position
        targetPos    = parsed_features['targetPos'][0:2]
        
        
        # num_Target
        numTarget = parsed_features['numTarget']-1
        
        
        # label in case model with Quadrants
        if params.model_with_quadrants:
            circle_mapping = np.load('Params/circle.npy')
            angle_mapping = np.load('Params/angle_in_quadrant.npy')
            quadrant_mapping = np.load('Params/quadrants.npy')
            label_mapping = np.concatenate([circle_mapping,quadrant_mapping, angle_mapping], axis=1)
            label_mapping_tensor = tf.placeholder_with_default(label_mapping, [48,11])
        
        else:
            label_mapping_tensor = tf.placeholder_with_default(np.zeros((48,11)), [48,11])
            
        label     = label_mapping_tensor[numTarget]
    
        # delayTime
        delayTime = parsed_features['delayTime']
        
        # Preprocess target_pos
        return spikeRasters, isSuccessful, targetPos, numTarget, label, delayTime

    
    size_batch = params.batch_size
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(size_batch)
    
    iterator = dataset.make_initializable_iterator()
    
    inputs, isSuccessful, targetPos, numTarget, label, delayTime = iterator.get_next()

    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'neurons_activity': inputs,
        'numTarget': numTarget,
        'label': label,
        'targetPos': targetPos,
        'delayTime': delayTime,
        'isSuccessful': isSuccessful,
        
        'iterator_init_op': init_op,
    }

    return inputs
