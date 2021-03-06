import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import ipdb
import csv
from sketch_dec import SketchRNNDecoder
from draw_utils import plot_stroke

sketch_vec = "owl_z_tt.csv"    #shape 100,128
photo_vec = "photo_z_tt.csv"   # shape 100,7,7,160
MODEL_DIR = "./linreg_log/duo_set/b20l1-e5E100k-3L20010050/"

STEPS = 100000  # number of training batch-iteration
BATCH_SIZE = 20
LR = 0.00001  # learning rate
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 100
LOG_STEP_COUNT_STEPS = 1000

# train or generate
MODE = 'train'

def linreg_fn(features, labels, mode, params):
    """ defines forward prop, loss, summary ops, and train_op. """
    # input
    inp = features['x']
    # defines hidden layers?
    for units in params.get('hidden_units',[200,100,50]):
        inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.relu)
        # inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.sigmoid)
        inp = tf.layers.dropout(inp)
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

def get_pseuinv(arr):
    for i in range(len(arr)):
        x=arr[i]
        arr[i] = np.transpose(x)*(np.power(np.matmul(x,np.transpose(x)),-1))
        # arr[i]=np.linalg.pinv([x])
    return arr

"""load from npz""" #separate
# npz_dir = "./dataset/"
# pig = np.load(npz_dir+"pig.npz")
# elephant = np.load(npz_dir+"elephant.npz")

# # train
# train_input = np.append(elephant['train_photo'],pig['train_photo'],0)
# train_target = np.append(elephant['train_sketch'],pig['train_sketch'],0)

# # test
# test_input = np.append(elephant['test_photo'],pig['test_photo'],0)
# test_target = np.append(elephant['test_sketch'],pig['test_sketch'],0) 

"""load from npz""" #combined
npz_dir = "./dataset/"
elepig = np.load(npz_dir+"elephantpig.npz")

# train
train_input = elepig['train_photo']
train_target = elepig['train_sketch']
# ipdb.set_trace()
# test
test_input = elepig['test_photo']
test_target = elepig['test_sketch']

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
            # 'hidden_units':[200,100,50],
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
    inputs, targets = test_input, test_target #from npz
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
    # decoder = SketchRNNDecoder("/tmp/sketch_rnn/models/owl/lstm_test/")
    decoder = SketchRNNDecoder("/home/kelvin/sketchrnn_pretrained/elephantpig/lstm_test/")

    
    """Reconstruction for direct input case"""
    strokes = []
    for pred in preds:
        strokes.append(decoder.draw_from_z(np.expand_dims(pred['sketch_vector'],0)))
    
    """Reconstruction for pseudo inverse case"""
    # vec = []
    # for pred in preds:
    #     vec.append(pred['sketch_vector'])    
    # vec=get_pseuinv(vec)
    # # ipdb.set_trace()
    # for i in range(len(vec)):
    #     strokes.append(decoder.draw_from_z(np.expand_dims(vec[i],0)))

    """Reconstruction for target data"""
    strokes_tar = []
    # inp, targ = get_csv_dataset()
    # ipdb.set_trace()
    for i in range(len(targets)):
        strokes_tar.append(decoder.draw_from_z(np.expand_dims(targets[i],0)))
    
    fig = plt.figure()
    N = 9
    gs = gridspec.GridSpec(2, N)
    
    for n in range(N):
        ax = fig.add_subplot(gs[0, n])
        plot_stroke(ax, strokes[n])
        ax = fig.add_subplot(gs[1, n])
        plot_stroke(ax, strokes_tar[n])

    # for i in range(1):
    #     for j in range(5):
    #         ax = fig.add_subplot(gs[i, j])
    #         plot_stroke(ax, strokes[c])
    #         c+=1

    plt.show()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    if MODE == 'train':
        tf.app.run(main=train)
    elif MODE == 'generate':
        tf.app.run(main=gen)

#DEFAULT PARAMS: batch 20, epoch 100000, LR 0.0001    