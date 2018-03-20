"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf
import numpy as np

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_dir is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    
    # Get paths for datasets
#    train_filenames = ['train-%.2d-of-%.2d.tfrecords' % (
#            shard, params.num_train_shards) for shard in range(params.num_train_shards)]
#    dev_filenames   = ['dev-%.2d-of-%.2d.tfrecords' % (
#            shard, params.num_dev_shards) for shard in range(params.num_dev_shards)]
#    test_filenames  = ['test-%.2d-of-%.2d.tfrecords' % (
#            shard, params.num_test_shards) for shard in range(params.num_test_shards)]

    if params.choose_target:
        params.train_size = np.sum(np.array(params.len_train_by_target)[params.target])
        params.eval_size  = 10*params.num_classes
        pass
    else:
        params.target = range(params.num_classes)
        
    train_filenames = ['train{}.tfrecords'.format(targ+1) for targ in params.target]
    dev_filenames   = ['dev{}.tfrecords'.format(targ+1) for targ in params.target]
#    test_filenames  = ['test{}.tfrecords'.format(targ+1) for targ in params.target]
    
    train_files = [os.path.join(args.data_dir,data_file) for data_file in train_filenames]
    dev_files = [os.path.join(args.data_dir,data_file) for data_file in dev_filenames]

    print(train_files, params.train_size, params.num_classes)
    # Create the input data pipeline
    logging.info("Creating the datasets...")
 

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_files, params)
    eval_inputs  = input_fn('eval' , dev_files  , params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)