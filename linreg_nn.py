import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import ipdb
import csv
from sketch_dec import SketchRNNDecoder,movie
from draw_utils import plot_stroke, to_abs, init_ax
from sklearn import decomposition
from itertools import chain

# sketch_vec = "owl_z_tt.csv"    #shape 100,128
# photo_vec = "photo_z_tt.csv"   # shape 100,7,7,160
# MODEL_DIR = "./linreg_log/owl-3L/b20l1e-4E100k/"
MODEL_DIR = "./linreg_log/duo_set/DEFAULT/"


STEPS = 100000  # number of training batch-iteration
BATCH_SIZE = 20
LR = 0.0001  # learning rate
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 100
LOG_STEP_COUNT_STEPS = 1000

# train or generate
MODE = 'generate'

def linreg_fn(features, labels, mode, params):
    """ defines forward prop, loss, summary ops, and train_op. """
    # input
    inp = features['x']
    # defines hidden layers?
    for units in params.get('hidden_units',[20]):
        inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.relu, name='feat_input')
        # inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.sigmoid)
        # inp = tf.layers.dropout(inp)
    # predictions
    preds = tf.layers.dense(inputs=inp, units=128, name="predictions")

    if mode == tf.estimator.ModeKeys.PREDICT:
        assignment_map = {}
        tf.train.init_from_checkpoint(MODEL_DIR, assignment_map)
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"sketch_vector": preds})

    avg_loss = tf.losses.mean_squared_error(labels, preds)

    # summaries to be shown in tensorboard
    tf.summary.scalar('train_loss', avg_loss)

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * avg_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {"rmse": avg_loss}
        return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)
    
    # error if it is not train mode
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params.get("learning_rate", None))
    # optimizer = tf.train.FtrlOptimizer(params.get("learning_rate", None))

    train_op = optimizer.minimize(loss=avg_loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
    
def get_fake_dataset():
    """ create fake input and target data. """
    input_data_shape = [7, 7, 160]
    target_data_shape = [128]
    num_data = 10  # size of data

    x = np.random.random([num_data] + input_data_shape).reshape(num_data, 7 * 7 * 160)
    t = np.random.random([num_data] + target_data_shape)
    return x, t

def csv_parse(f):
    data = []
    with open(f,'rb') as cf:
        rd = csv.reader(cf, delimiter = ',')
        for row in rd:
            data.append(map(float,row))
    return np.asarray(data)

def get_csv_dataset():
    x_raw = csv_parse(photo_vec)
    targ = csv_parse(sketch_vec)
    return x_raw, targ

def vis_clusters(inputs,targets):
    
    x, y = inputs, targets

    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x = pca.transform(x)
    colors = ['turquoise','darkorange']


    plt.scatter(x[:95,0],x[:95,1],c='turquoise')
    plt.scatter(x[95:,0],x[95:,1],c='darkorange')
    plt.show()

"""load from npz""" #combined
npz_dir = "./dataset/"
elepig = np.load(npz_dir+"elephantpig.npz")

# train
train_input = elepig['train_photo']
train_target = elepig['train_sketch']

# test
test_input = elepig['test_photo']
test_target = elepig['test_sketch']

# """load from owl.npz""" #combined
# npz_dir = "./dataset/"
# owl = np.load(npz_dir+"owl.npz")

# # train
# train_input = owl['train_photo']
# train_target = owl['train_sketch']

# # test
# test_input = owl['test_photo']
# test_target = owl['test_sketch']

def train(arg):
    """build & train"""
    
    # create a model_dir
    tf.gfile.MakeDirs(MODEL_DIR)

    # generate fake dataset as numpy arrys.
    # TODO: this should be replaced by csv_parse()
    # inputs, targets = get_fake_dataset() #random data
    
    # inputs, targets = get_csv_dataset() #from csv
    # inputs = inputs[:-5]
    # targets = targets[:-5]
    # inputs = get_pseuinv(inputs) #obtain pseudo inverse of input
    inputs, targets = train_input, train_target #from npz

    # input_fn to feed the data to an estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': inputs},
        y=targets,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_epochs=None
    )

    # define type and shape of the input data
    my_feature_columns = [tf.feature_column.numeric_column(
        key="x",
        shape=[7 * 7 * 160]
    )]    

    my_config = tf.estimator.RunConfig(
        model_dir=MODEL_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        keep_checkpoint_max=None,
        log_step_count_steps=LOG_STEP_COUNT_STEPS,
        )

    # create a linear regression model
    model = tf.estimator.Estimator(
        model_fn=linreg_fn,
        params={
            'feature_columns': my_feature_columns,
            'learning_rate': LR,
        },
        config=my_config
    )
    
    # start the training
    model.train(
        input_fn=train_input_fn,
        steps=STEPS)

def gen(arg):
    """
    Load trained weight of linreg and sketch-rnn decoder, and 
    generate sketch results.
    """
    # put a log text
    tf.logging.info("generation mode.")

    # generate fake dataset as numpy arrys.
    # TODO: this should be replaced by csv_parse()
    # inputs, targets = get_fake_dataset() #random data
    # inputs, targets = get_csv_dataset()# from csv
    # test_in = inputs[-5:]
    # test_tar = targets[-5:]
    # inputs,targets = test_in, test_tar
    inputs, targets = train_input, train_target #from npz
    # ipdb.set_trace()
    # define type and shape of the input data
    my_feature_columns = [tf.feature_column.numeric_column(
        key="x",
        shape=[7 * 7 * 160]
    )]    

    my_config = tf.estimator.RunConfig(model_dir=MODEL_DIR)

    # create a linear regression model
    model = tf.estimator.Estimator(
        model_fn=linreg_fn,
        params={
            'feature_columns': my_feature_columns,
        },
        config=my_config
    )

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': inputs},
        y=targets,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_epochs=1
    )
    preds = model.predict(input_fn=pred_input_fn)

    # for pred in preds:
    #     # pred['sketch_vector'] is numpy.array
    #     print(pred['sketch_vector'].shape)
    
    # ipdb.set_trace()
    # decoder = SketchRNNDecoder("/home/kelvin/OgataLab/sketchrnn_pretrained/owl/lstm_test/")
    decoder = SketchRNNDecoder("/home/kelvin/OgataLab/sketchrnn_pretrained/elephantpig/lstm_test/")

    
    """Reconstruction for direct input case"""

    import matplotlib.animation as anim

    # init decoder
    # decoder = SketchRNNDecoder(args.log_dir, args.gpu_mode)
    targ_s = []
    for i in range(len(targets)):
        targ_s.append(decoder.draw_from_z(np.expand_dims(targets[i],0)))
    
    # # get a drawing sequence, and convert stroke-3 format data to list of x-y
    # s = []
    # for pred in preds:
    #     s.append(decoder.draw_from_z(np.expand_dims(pred['sketch_vector'],0)))
    cluster_data=list(preds)
    # for i in preds:
    #     cluster_data.append(i['sketch_vector'])
    ipdb.set_trace()

    def sav2gif(sketch_arr, f_name):
        x, y = to_abs(sketch_arr)

        # initialize figure
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Generated Sequence')

        # plot a stroke sequence as movie and static plot
        ax = fig.add_subplot(121)
        plot_stroke(ax, sketch_arr)

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
        filename = f_name

        # gif file output requires imagemagick.
        # mp4 requires ffmpeg.
        ani.save(filename, writer='imagemagick')

        # show the created movie.
        plt.show()
    # ipdb.set_trace()
    vis_clusters(cluster_data,targets)
    # sav2gif(targ_s[102], '1lay_trntargelepig103_z.gif')
    # sav2gif(s[102], '1lay_trnpredelepig103_z.gif')

    # strokes = []
    # for pred in preds:
    #     # ipdb.set_trace()
    #     strokes.append(decoder.draw_from_z(np.expand_dims(pred['sketch_vector'],0)))

    # # """Reconstruction for target data"""
    # strokes_tar = []
    
    # for i in range(len(targets)):
    #     strokes_tar.append(decoder.draw_from_z(np.expand_dims(targets[i],0)))
    
    # fig = plt.figure()
    # N = 5
    # gs = gridspec.GridSpec(2, N)
    
    # for n in range(N):
    #     strokes(n).movie()
    #     ax = fig.add_subplot(gs[0, n])
    #     plot_stroke(ax, strokes[n])
    #     ax = fig.add_subplot(gs[1, n])
    #     plot_stroke(ax, strokes_tar[n])

    # # for i in range(1):
    # #     for j in range(5):
    # #         ax = fig.add_subplot(gs[i, j])
    # #         plot_stroke(ax, strokes[c])
    # #         c+=1

    # plt.show()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    if MODE == 'train':
        tf.app.run(main=train)
    elif MODE == 'generate':
        tf.app.run(main=gen)

#DEFAULT PARAMS: batch 20, epoch 100000, LR 0.0001    