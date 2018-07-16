from sketch_dec import SketchRNNDecoder
from draw_utils import plot_stroke
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import argparse
import ipdb

dir = "./dataset/"
data = np.load(dir+"elephantpig.npz")
# data = np.load(dir+"owl.npz")

# training data
train_inputs, train_targets = data['train_photo'], data['train_sketch']

# test data
test_inputs, test_targets = data['test_photo'], data['test_sketch']

def rmse(target,inputs,A):
    '''Show RMSE'''
    rms = 0
    data_no = len(target)
    for i in range(data_no):
        pred = np.matmul(inputs[i],A)
        rms += ((target[i] - pred)**2).mean()
    rms = np.sqrt(rms/data_no)
    print("loss: "+str(rms))

def pseu_inv(x):
    return np.linalg.pinv(x)

def pseuinv_w_lamda(x,n,constant):
    # ipdb.set_trace()
    s_1 = (np.matmul(x,np.transpose(x))/n)+constant*np.identity(n)
    s_2 = np.power(s_1,-1)
    s_3 = np.matmul((np.transpose(x)/n),s_2)
    return s_3

def gen_train(n,m,A):
    # define input
    x = train_inputs[n:m]

    # predictions
    preds = []
    for i in range(len(x)):
        preds.append(np.matmul(x[i],A))
    
    return preds

def gen_test(n,m, A):
    # define input
    x = test_inputs[n:m]

    # predictions
    preds = []
    for i in range(len(x)):
        preds.append(np.matmul(x[i],A))
    
    return preds

def main():
    '''Linear Regression'''
    # specify mode
    c = 'test'
    # define x and Y
    x, Y = train_inputs, train_targets
    
    # total number of data
    data_no = len(x)

    # # deriving A
    # A = np.matmul(pseu_inv(x),Y)
    # # ipdb.set_trace()

    # derive A with constant k
    k = 0.0001
    n = len(train_inputs)
    A = np.matmul(pseuinv_w_lamda(x,n,k),Y)

    # ipdb.set_trace()

    # root mean squared error
    rmse(Y,x,A)
    
    # call sketch-rnn decoder
    decoder = SketchRNNDecoder("/home/kelvin/OgataLab/sketchrnn_pretrained/elephantpig/lstm_test/")
    # decoder = SketchRNNDecoder("/home/kelvin/OgataLab/sketchrnn_pretrained/owl/lstm_test/")

    
    def decode(c):

        if c=='train':
            # # generate predictions from training data, 2 category
            # pred_train = np.vstack((gen_train(0,3,A),gen_train(101,103,A)))
            # data_train = np.vstack((train_targets[:3],train_targets[101:103]))
            # input_train = np.vstack((train_inputs[:3],train_inputs[101:103]))
            
            # generate predictions from training data, 2 category
            pred_train = np.vstack((gen_train(0,2,A),gen_train(56,59,A)))
            data_train = np.vstack((train_targets[:2],train_targets[56:59]))
            input_train = np.vstack((train_inputs[:2],train_inputs[56:59]))

            # Validation of A
            rmse(data_train,input_train,A)

            # decoding target vectors
            targ_train = []
            for p in range(len(data_train)):
                targ_train.append(decoder.draw_from_z(np.expand_dims(data_train[p],0)))
            # decoding predictions from training set
            strokes_train = []
            for p in range(len(pred_train)):
                strokes_train.append(decoder.draw_from_z(np.expand_dims(pred_train[p],0)))
            
            # plotting training target and predictions
            fig = plt.figure()
            N = 5
            gs = gridspec.GridSpec(2, N)
            
            for n in range(N):
                ax = fig.add_subplot(gs[0, n])
                plot_stroke(ax, targ_train[n])
                ax = fig.add_subplot(gs[1, n])
                plot_stroke(ax, strokes_train[n])          
            plt.show()

        elif c=='test':
            # # generate sample predictions from training data
            # pred_test = np.vstack((gen_test(0, 3, A),gen_test(5, 9, A)))
            # data_test = np.vstack((test_targets[:3],test_targets[5:9]))
            # input_test = np.vstack((test_inputs[:3],test_inputs[5:9]))

            # generate predictions from training data
            pred_test = gen_test(0, 10, A)
            data_test = test_targets
            input_test = test_inputs

            # validation of A
            rmse(data_test,input_test,A)

            # decoding target vectors
            targ_test = []
            for p in range(len(data_test)):
                targ_test.append(decoder.draw_from_z(np.expand_dims(data_test[p],0)))
            
            # decoding predictions from training set
            strokes_test = []
            for p in range(len(pred_test)):
                strokes_test.append(decoder.draw_from_z(np.expand_dims(pred_test[p],0)))
            
            # plotting training target and predictions
            fig = plt.figure()
            N = 9
            gs = gridspec.GridSpec(2, N)
            
            for n in range(N):
                ax = fig.add_subplot(gs[0, n])
                plot_stroke(ax, targ_test[n])
                ax = fig.add_subplot(gs[1, n])
                plot_stroke(ax, strokes_test[n])          
            plt.show()

    #plotting test target and predictions
    decode(c)

if __name__ == "__main__":
    main()