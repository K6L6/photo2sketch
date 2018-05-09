#!/bin/bash

hps=""
hps+="dataset=airplane.npz,"
hps+="num_steps=100000,"
hps+="dec_rnn_size=256,"
hps+="dec_model=layer_norm,"
hps+="enc_rnn_size=128,"
hps+="enc_model=layer_norm,"
hps+="z_size=32,"
hps+="batch_size=30,"

log=log/airplane_1
data=../data

python sketch_rnn_train.py --log_root=$log --data_dir=$data $hps