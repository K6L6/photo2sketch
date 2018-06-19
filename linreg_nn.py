import tensorflow as tf
import numpy as np
import ipdb
import csv

sketch_vec = "owl_z.csv"    #shape 100,128
photo_vec = "photo_z.csv"   # shape 100,7,7,160

def csv_parse(f):
    data = []
    with open(f,'rb') as cf:
        rd = csv.reader(cf, delimiter = ',')
        for row in rd:
            data.append(map(float,row))
    return np.asarray(data)

x_raw = csv_parse(photo_vec)
targets = csv_parse(sketch_vec)
dataset = []
for i in range(len(x_raw)):
    dataset.append((x_raw[i],targets[i]))

# dataset = np.asarray(dataset) 

def linreg_fn(x,y,mode,params):
    inp = tf.feature_column.input_layer(x, params["feature_columns"])

    for units in params.get("hidden_units",[20]):
        inp = tf.layers.dense(inputs=inp,units=units, activation=tf.nn.relu)
    
    out_layer = tf.layers.dense(inputs=inp,units=1)

    preds = tf.squeeze(out_layer,1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"sketch_vector":preds})
    
    avg_loss = tf.losses.mean_squared_error(y,preds)

    batch_size = tf.shape(y)[0]
    total_loss = tf.to_float(batch_size)*avg_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer",tf.trian.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate",None))
        trian_op = optimizer.minimize(loss=avg_loss,global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,trian_op=train_op)
    
    rmse = tf.metrics.root_mean_squared_error(y, preds)
    
    eval_metrics = {"rmse": rmse}
    
    return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)

def main(arg):
    """build & train"""
    assert len(arg) == 1
    
    (train) = dataset
    
    def norm(x,y):
        return x,y

    train = train.map(norm)

    def inp_train():
        return(train.shuffle(1000).batch(50).repeat().make_one_shot_iterator().get_next())

    feature_columns = [tf.feature_column.numeric_column(key="x")]

    model = tf.estimator.Estimator(model_fn=linreg_fn, params={"feature_columns":feature_columns, "learning_rate":0.001,"optimizer":tf.train.AdamOptimizer,"hidden_units":[20,20]})

    model.train(input_fn=inp_train, steps=STEPS)
    # linreg_fn(x,y,mode = tf.estimator.ModeKeys.TRAIN,params={"learning_rate":0.001,"optimizer":tf.train.AdamOptimizer,"hidden_units":[20,20]})

    print("\n"+80*"*")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)