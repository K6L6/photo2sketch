from sketch_dec import SketchRNNDecoder
from draw_utils import plot_stroke, to_abs, init_ax
from sklearn import decomposition
from itertools import chain
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim
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

def vis_clusters(inputs,targets):
    
    x, y = inputs, targets

    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x = pca.transform(x)
    colors = ['turquoise','darkorange']


    plt.scatter(x[:95,0],x[:95,1],c='turquoise')
    plt.scatter(x[95:,0],x[95:,1],c='darkorange')
    plt.show()

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
    return (np.linalg.inv((np.transpose(x).dot(x))+np.identity(7840)*0.1)).dot(np.transpose(x))
    # return np.linalg.pinv(x)

def pseuinv_w_lamda(x,n,constant):
    # ipdb.set_trace()
    s_1 = (np.matmul(np.transpose(x),x)/n)+constant*np.identity(n)
    s_2 = np.linalg.inv(s_1)
    s_3 = np.matmul(s_2,(np.transpose(x)/n))
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

def main():
    '''Linear Regression'''
    # specify mode
    c = 'train'
    # define x and Y
    x, Y = train_inputs, train_targets
    
    # total number of data
    data_no = len(x)

    # deriving A
    A = np.matmul(pseu_inv(x),Y)
    

    # # derive A with constant k
    # k = 0.001
    # n = len(train_inputs[0])
    # A = np.matmul(pseuinv_w_lamda(x,n,k),Y)

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
            
            # generate predictions from training data, 1 category
            pred_train = np.vstack((gen_train(0,2,A),gen_train(56,59,A)))
            data_train = np.vstack((train_targets[:2],train_targets[56:59]))
            input_train = np.vstack((train_inputs[:2],train_inputs[56:59]))

            pred_cluster = gen_train(0,190,A)
            targ_cluster = train_targets
            
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
            
            vis_clusters(pred_cluster,targ_cluster)

            # sav2gif(strokes_train[3], 'trn_pred_elepig102.gif')
            
            # # plotting training target and predictions
            # fig = plt.figure()
            # N = 5
            # gs = gridspec.GridSpec(2, N)
            
            # for n in range(N):
            #     ax = fig.add_subplot(gs[0, n])
            #     plot_stroke(ax, targ_train[n])
            #     ax = fig.add_subplot(gs[1, n])
            #     plot_stroke(ax, strokes_train[n])          
            # plt.show()

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
            
            sav2gif(strokes_test[5], 'tst_pred_elepig6.gif')
            # # plotting training target and predictions
            # fig = plt.figure()
            # N = 9
            # gs = gridspec.GridSpec(2, N)
            
            # for n in range(N):
            #     ax = fig.add_subplot(gs[0, n])
            #     plot_stroke(ax, targ_test[n])
            #     ax = fig.add_subplot(gs[1, n])
            #     plot_stroke(ax, strokes_test[n])          
            # plt.show()

    #plotting test target and predictions
    decode(c)

if __name__ == "__main__":
    main()