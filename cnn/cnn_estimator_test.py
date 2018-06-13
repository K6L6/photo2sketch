import ipdb
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os

parser = argparse.ArgumentParser(description="test")
data_path = os.path.join(
    os.path.expanduser('~'),
    'data'
)    
parser.add_argument('--data_path', default=data_path, help='path to save data. default is ~/data.')
parser.add_argument('--batch_size', type=int, default=5, help='size of batch')
parser.add_argument('--steps', type=int, default=10, help='max number of training batch iteration')
parser.add_argument('--save_every', type=int, default=10, help='interval of saving per step')
parser.add_argument('--save_max', type=int, default=5, help='number of maximum checkpoints')
parser.add_argument('--log_every', type=int, default=2, help='interval of logging per step')
parser.add_argument('--summ_every', type=int, default=2, help='interval of recording summary per step')
parser.add_argument('--model_dir', default='log/test', help='directory to put training log')

def get_data_as_np(mnist):
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels

    # reshape inputs to 4D array
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    return train_x, train_y, test_x, test_y

def model_fn(features, labels, mode, params):
    """ define my model """
    dp_keep_prob = 0.8

    # get tensor input
    h = features['x']

    h = tf.layers.conv2d(
        h,
        filters=4,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.tanh
    )
    h = tf.nn.dropout(h, dp_keep_prob)

    h = tf.layers.conv2d(
        h,
        filters=4,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.tanh
    )
    h = tf.nn.dropout(h, dp_keep_prob)

    h = tf.layers.flatten(h)

    y = tf.layers.dense(h, 10, activation=tf.tanh)

    # returns pred
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y': y
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # get loss
    loss = tf.losses.mean_squared_error(y, labels)

    # summaries to be shown in tensorboard
    tf.summary.scalar('train_loss', loss)

    # mode for evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)

    # create train_op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(argv):
    # parse arguments
    args = parser.parse_args(argv[1:])

    # create log directory
    tf.gfile.MakeDirs(args.model_dir)

    # maybe download mnist data
    # data is numpy.arrays
    # we use only input data because the model is auto-encoder
    train_x, train_y, test_x, test_y = get_data_as_np(
        input_data.read_data_sets(args.data_path, one_hot=True))

    # define type of input data
    my_feature_columns = [tf.feature_column.numeric_column(
        key="x",
        shape=[28, 28, 1]   # width=28, height=28, channel=1
    )]

    # configuration of the model
    my_config = tf.estimator.RunConfig(
        save_checkpoints_steps = args.save_every,
        keep_checkpoint_max = args.save_max,
        log_step_count_steps = args.log_every,
        save_summary_steps = args.summ_every,
    )

    # create model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': my_feature_columns
        },
        model_dir=args.model_dir,
        config=my_config)

    # create input pipeline for training.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        batch_size=args.batch_size,
        shuffle=True, 
        num_epochs=3
    )

    # create input pipeline for evaluation.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        y=test_y,
        batch_size=args.batch_size,
        shuffle=True, 
        num_epochs=1
    )

    # start the training. 
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # get prediction after training.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x[:args.batch_size]},
        y=test_y[:args.batch_size],
        batch_size=args.batch_size,
        shuffle=False
    )

    # predict() returns generator
    preds = model.predict(
        input_fn=test_input_fn,
    )

    # this is numpy array
    for pred in preds:
        print(pred['y'])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
