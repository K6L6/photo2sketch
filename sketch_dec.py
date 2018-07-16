import ipdb
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import json
from itertools import chain

from sketch_rnn.model import Model, sample
from sketch_rnn.utils import to_normal_strokes
from draw_utils import plot_stroke, to_abs, init_ax

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Argument parser to specify log_dir by the arguments.
parser = argparse.ArgumentParser(description="sketch-rnn decoder test")
parser.add_argument('plot_mode', type=str, help='Mode to show decode results. movie or plot.')
parser.add_argument('log_dir', type=str, help='Log directory of sketch-rnn')
parser.add_argument('--gpu_mode', type=bool, default=False, help='Using gpu or not.')
parser.add_argument('--savename', type=str, help='name of the figure. if None, the default name will be used.')

class SketchRNNDecoder(object):
    """ Helper class to acquire sketch-rnn's decoder outputs from z as numpy.array. 
        Sketch-rnn: https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn
    
    Args:
        log_dir (String): Path of a directory which has ckeckpoint data of sketch-rnn. 
                        The checkpoint example can be found in http://download.magenta.tensorflow.org/models/sketch_rnn.zip .
        gpu_mode (Boolean): Use GPUs or not. Default mode is False (CPU-mode.)

    Usage:
        1. Initialize decoder
        decoder = SketchRNNDecoder('<path-to-log-dir>')
        
        2. Call draw_from_z to get drawing sequence, with random z.
        sequence = decoder.draw_from_z()

        3. Or you can input z as np.array, and manual temperature.
        # z = np.random.randn(1, decoder.get_z_size()) # faked z by random.
        sequence = decoder.draw_from_z(z, 0.8)
    """
    def __init__(self, log_dir, gpu_mode=False):
        """ Initialize sketch-rnn model as a VAE-RNN """
        self.log_dir = log_dir

        # load hyper-parameters from log_dir.
        self.hps = self.get_hps(self.log_dir)
        
        # store the length of sequence for plotting
        self.seq_len_plot = self.hps.max_seq_len
        
        # manually set params for inference-mode
        self.hps.is_training = 0
        self.hps.batch_size = 1
        self.hps.use_recurrent_dropout = 0
        self.hps.use_output_dropout = 0
        self.hps.max_seq_len = 1

        # define tensors of sketch-rnn.
        self.model = Model(self.hps, gpu_mode)

        # initialize session.
        self.sess = tf.InteractiveSession()        

        # restore the trained weight of a sketch-rnn from log_dir.
        self.restore(self.log_dir)

    def draw_from_z(self, z=None, temperature=0.2, greedy_mode=False):
        """ get stroke-3 format drawing sequence from z as a numpy arrray. 
            Note that the size of z is given by get_z_size().
            args:
            z: numpy array. an input latent vector. If z = None, a random value will be used.
            temperature: float. randomness to take one from the predicted component.
            greedy_mode: bool. if true, the rnn always takes the componenet that has the biggest mixture ratio.
        """
        strokes, mixture_params = sample(self.sess, 
            self.model, 
            seq_len=self.seq_len_plot, 
            z=z, 
            temperature=temperature,
            greedy_mode=greedy_mode)
        return to_normal_strokes(strokes)

    def get_z_size(self):
        """ get the dimensional size of sketch-rnn's z (latent representation) """
        return self.hps.z_size

    def restore(self, log_dir):
        """ restore variables from log_dir """
        rnn_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='vector_rnn'
        )
        saver = tf.train.Saver(rnn_vars)
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def get_hps(self, log_dir):
        """ load hps from json file """
        with open(os.path.join(log_dir, 'model_config.json'), 'r') as f:
            hps_json = json.load(f)
        hps = tf.contrib.training.HParams()
        for key, value in hps_json.items():
            hps.add_hparam(key, value)
        return hps
    
def sketch_rnn_decode(argv):
    """ test of SketchRNNDecoder. """
    # # parser arguments
    args = parser.parse_args(argv[1:])

    if args.plot_mode == 'plot':
        plot(args)
    elif args.plot_mode == 'movie':
        movie(args)
    else:
        raise ValueError('plot_mode should be plot or movie')


def movie(args):
    import matplotlib.animation as anim

    # init decoder
    decoder = SketchRNNDecoder(args.log_dir, args.gpu_mode)

    # get a drawing sequence, and convert stroke-3 format data to list of x-y
    s = decoder.draw_from_z()
    x, y = to_abs(s)

    # initialize figure
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Generated Sequence')

    # plot a stroke sequence as movie and static plot
    ax = fig.add_subplot(121)
    plot_stroke(ax, s)

    # axes for the movie
    ax = fig.add_subplot(122)
    init_ax(ax, x, y)

    # get flagments to finish a line or not
    def get_p(abs_x):
        t = 0
        res = []
        for _x in x:
            t += len(_x)
            res.append(t)
        return res

    p = get_p(x)

    # flatten lists
    x = list(chain.from_iterable(x))
    y = list(chain.from_iterable(y))

    # define a frame func to update the figure at each step
    def frame(t):
        if not t + 1 in p:
            ax.plot([x[t], x[t+1]], [y[t], y[t+1]], color='black', lw=2.0)
    
    # create an animation using the defined frame func.
    # interval means an time interval between frames in milli-seconds
    ani = anim.FuncAnimation(fig, frame, interval=75, frames=len(x) - 1)

    # set a filename
    if args.savename is None:
        filename = 'seq_random_z.gif'

    # gif file output requires imagemagick.
    # mp4 requires ffmpeg.
    ani.save(filename, writer='imagemagick')

    # show the created movie.
    plt.show()

def plot(args):

    # init decoder
    decoder = SketchRNNDecoder(args.log_dir, args.gpu_mode)

    # plot in 5 x 5 cells in a matplotlib figure
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(5, 5)

    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            plot_stroke(ax, decoder.draw_from_z())

    plt.show()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(sketch_rnn_decode)