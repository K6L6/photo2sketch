import tensorflow as tf
import numpy as np
import ipdb
import csv

sketch_vec = "owl_z.csv"    #shape 100,128
photo_vec = "photo_z.csv"   # shape 100,7,7,160
# model_dir="/home/kelvin/OgataLab/sketch-wmultiple-tags/linreg_log/"
model_dir = "./linreg_log/test1"

STEPS = 1000000  # number of training batch-iteration
BATCH_SIZE = 5
LR = 0.0001  # learning rate

def csv_parse(f):
    data = []
    with open(f,'rb') as cf:
        rd = csv.reader(cf, delimiter = ',')
        for row in rd:
            data.append(map(float,row))
    return np.asarray(data)

x_raw = csv_parse(photo_vec)
targ = csv_parse(sketch_vec)
# dataset = []
# for i in range(len(x_raw)):
#     dataset.append((x_raw[i],targets[i]))

    # dataset = np.asarray(dataset) 

def linreg_fn(features, labels, mode, params):
    """ defines forward prop, loss, summary ops, and train_op. """

    inp = features['x']

    for units in params.get("hidden_units",[20]):
        inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.relu,name="feat_input")
    
    preds = tf.layers.dense(inputs=inp, units=128, name="predictions")

    avg_loss = tf.losses.mean_squared_error(labels, preds)

    # summaries to be shown in tensorboard
    tf.summary.scalar('train_loss', avg_loss)

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * avg_loss

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"sketch_vector": preds})

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {"rmse": avg_loss}
        return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)
    
    # error if it is not train mode
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params.get("learning_rate", None))
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

def main(arg):
    """build & train"""
    
    # create a model_dir
    tf.gfile.MakeDirs(model_dir)

    # generate fake dataset as numpy arrys.
    # TODO: this should be replaced by csv_parse()
    # inputs, targets = get_fake_dataset()
    inputs, targets = x_raw, targ
    
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
        model_dir=model_dir,
        save_summary_steps=50,
        save_checkpoints_steps=100,
        keep_checkpoint_max=None,
        log_step_count_steps=50,
        )

    # create a linear regression model
    model = tf.estimator.Estimator(
        model_fn=linreg_fn,
        params={
            'feature_columns': my_feature_columns,
            'learning_rate': LR
        },
        config=my_config
    )
    
    # start the training
    model.train(
        input_fn=train_input_fn,
        steps=STEPS)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)