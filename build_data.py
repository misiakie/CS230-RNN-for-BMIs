"""Read, split and convert our Matlab into tfrecords"""

import os
import sys
import tensorflow as tf
import scipy.sparse as sparse
import random
import numpy as np
from scipy.io import loadmat

from build_data_params import *

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def from_sparse_matrix_to_feature(features, mapping_number,name="default"):
    sparse_array = features[mapping_number]
    index_x, index_y, values = sparse.find(sparse_array)
    shape = sparse_array.shape
    
    feature = {    name+'_x': _int64_feature(index_x), 
                   name+'_y': _int64_feature(index_y),
                   name+'_values': _float_feature(values),
                   name+'_shape':  _int64_feature(shape)
              }
    return feature 

def from_int_matrix_to_feature(features, mapping_number, name="default"):
    return {name: _int64_feature(np.array(features[mapping_number].flatten()))}

def from_float_matrix_to_feature(features, mapping_number, name="default"):
    return {name: _float_feature(np.array(features[mapping_number].flatten()))}

def from_string_matrix_to_feature(features, mapping_number, name="default"):
    return {name: _bytes_feature(np.array(features[mapping_number].flatten(), dtype='str'))}

def get_target_position(features, mapping_number=11, mapping_param_number=15, name="targetPos"):
    return {name: _float_feature(np.array(features[mapping_number][0,0][mapping_param_number]).flatten())}

def create_example_from_mat_file(file_addrs):
    """
    Converts a mat file containing one experiment into a tf Example
    
    Args:
        file_addrs: address of the file containing the Matlab structure
             
    Returns:
        tf Example reprensenting this experiment
    """
    mat = loadmat(file_addrs)
    features = list(mat['structarr'][0,0])

    # Convert the features 
    feature = {}
    for k in map_name_to_type_and_position:
        map_number, map_type = map_name_to_type_and_position[k]
        try:
            if map_type == int:
                feature.update(from_int_matrix_to_feature(features, map_number, name=k))
            if map_type == float:
                feature.update(from_float_matrix_to_feature(features, map_number, name=k))
            if map_type == 'str':
                feature.update(from_string_matrix_to_feature(features, map_number, name=k))
            if map_type == 'sparse':
                feature.update(from_sparse_matrix_to_feature(features, map_number, name=k))
        except:
            print(k,file_addrs)

    feature.update(get_target_position(features))

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def write_dataset(filename_list, output_directory, num_threads=1, max_size=1000, name='train'):
    """
    Writes all the examples in filename_list in several .tfrecords files.
    
    Args:
        filename_list: list of the files containing the examples to convert into examples to include
        output_directory: name of the directory where the .tfrecords will be saved
        num_threads: NOT IMPLEMENTED
        max_size: maximum size for one .tfrecords file (a "shard")
        name : name of the .tfrecords file                 
    """
        
    dataset_size = len(filename_list)
    
    if (dataset_size%max_size):
        num_shards   = dataset_size // max_size + 1
    else:
        num_shards = dataset_size // max_size
        
    print("Number of shards for {} dataset: {}".format(name,num_shards))
    
    for shard in range(num_shards):
        # open the TFRecords file
        output_filename = '%s-%.2d-of-%.2d.tfrecords' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file) 
        
        # number of examples in this shard
        if shard<(num_shards-1)|(not(dataset_size%max_size)):
            shard_size = max_size
        else:
            shard_size = dataset_size - (num_shards-1)*max_size
        
        
        for i in range(shard_size):
            # print how many examples are saved every 100 images
            if not i % 100:
                print('{} Data: {}/{}'.format(name, shard*max_size+i+1, dataset_size))
                sys.stdout.flush()

            # Load the the .mat file and convert it to a tf example
            file_addrs = filename_list[max_size*shard+i]
            example    = create_example_from_mat_file(file_addrs)
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close() 


def create(filename_list, output_directory, 
           dev_size=500,test_size=500,shuffle=True, num_threads=1, max_size=1000):
    """
    Randomly shuffles the data and divides the examples between train, dev and test sets. 
    
    Args:
        filename_list: list of the files containing the examples to convert into examples to include
        output_directory: name of the directory where the .tfrecords will be saved
        dev_size: dev test size
        test_size: test set size
        num_threads: NOT IMPLEMENTED
        max_size: maximum size for one .tfrecords file (a "shard")
    """
        
    if shuffle:
        random.shuffle(filename_list)
        
    train_size = len(filename_list)-(dev_size+test_size)
    assert train_size > 0
    
    train_filenames = filename_list[:train_size]
    dev_filenames   = filename_list[train_size:train_size+dev_size]
    test_filenames  = filename_list[train_size+dev_size:]
    
    write_dataset(train_filenames, output_directory, num_threads, max_size, name='train')
    print("Train dataset completed")
    write_dataset(dev_filenames, output_directory, num_threads, max_size, name='dev')
    print("Dev dataset completed")
    write_dataset(test_filenames, output_directory, num_threads, max_size, name='test')
    print("Test dataset completed")
    

if __name__ == "__main__":    
    create(train_addrs, 'data')