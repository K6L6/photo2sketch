import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as grdspc
import sys
# from magenta.models.sketch_rnn import utils
from svg_parser import view_stroke3
import ipdb
sys.path.insert(0, './sketch_rnn/test')
import sketch_rnn_train

# ipdb.set_trace()

file_dir='/home/kelvin/OgataLab/magenta/magenta/models/sketch_rnn/sketchy_data/'
file1 = 'aaron_sheep.npz'
file2 = 'sheep250_ep3.npz'
strk_dat1x = []
strk_dat1y = []
strk_dat2x = []
strk_dat2y = []

data1 = np.load(file_dir+file1)
data2 = np.load(file_dir+file2)

train1 = data1['train']
for i in range(len(train1)):
    for j in range(len(train1[i])):
        strk_dat1x.append(train1[i][j][0])
        # ipdb.set_trace()
        strk_dat1y.append(train1[i][j][1])

train2 = data2['train']
for i in range(len(train2)):
    for j in range(len(train2[i])):
        strk_dat2x.append(train2[i][j][0])
        strk_dat2y.append(train2[i][j][1])

#stroke-3toline
# l = []
# for i in range(len(strk_dat2)):
#     l.append(utils.strokes_to_lines(strk_dat2[i]))
# print(l)
# for i in range(len(strk_dat1)):
#     l.append(utils.strokes_to_lines(strk_dat1[i]))
# print(l)

#2bigstrokes
# s5 = []
# for i in range(len(strk_dat2)):
#     s5.append(utils.to_big_strokes(strk_dat2[i]))
# print(s5)
# ipdb.set_trace()
"""Histogram"""
plt.subplot(221)
plt.hist(strk_dat1x, bins='auto')
plt.title("Aaron X")

plt.subplot(222)
plt.hist(strk_dat2x, bins='auto')
plt.title("Sketchy X")

plt.subplot(223)
plt.hist(strk_dat1y, bins='auto')
plt.title("Aaron Y")

plt.subplot(224)
plt.hist(strk_dat2y, bins='auto')
plt.title("Sketchy Y")

print("Aaronx = "+str(min(strk_dat1x))+","+str(max(strk_dat1x)))
print("sketchyx = "+str(min(strk_dat2x))+","+str(max(strk_dat2x)))
print("Aarony = "+str(min(strk_dat1y))+","+str(max(strk_dat1y)))
print("sketchyy = "+str(min(strk_dat2y))+","+str(max(strk_dat2y)))

data_dir = '/home/kelvin/OgataLab/magenta/magenta/models/sketch_rnn/sketchy_data/'
model_dir = '/tmp/sketch_rnn/models/owl/lstm/'
# model_dir = '/tmp/sketch_rnn/models/aaron_sheep/lstm/'
[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = sketch_rnn_train.load_env(data_dir, model_dir)

trainStrokesMinx = [np.concatenate([stroke[:,0] for stroke in train_set.strokes]).min()
,np.concatenate([stroke[:,0] for stroke in valid_set.strokes]).min()
,np.concatenate([stroke[:,0] for stroke in test_set.strokes]).min()]


trainStrokesMiny = [np.concatenate([stroke[:,1] for stroke in train_set.strokes]).min()
,np.concatenate([stroke[:,1] for stroke in valid_set.strokes]).min()
,np.concatenate([stroke[:,1] for stroke in test_set.strokes]).min()]


trainStrokesMaxx = [np.concatenate([stroke[:,0] for stroke in train_set.strokes]).max()
,np.concatenate([stroke[:,0] for stroke in valid_set.strokes]).max()
,np.concatenate([stroke[:,0] for stroke in test_set.strokes]).max()]


trainStrokesMaxy = [np.concatenate([stroke[:,1] for stroke in train_set.strokes]).max()
,np.concatenate([stroke[:,1] for stroke in valid_set.strokes]).max()
,np.concatenate([stroke[:,1] for stroke in test_set.strokes]).max()]


print(min(trainStrokesMinx), min(trainStrokesMiny), max(trainStrokesMaxx), max(trainStrokesMaxy))
# c=0
# row, col = 10, 10
# gs = grdspc.GridSpec(row, col)

# for i in range(row):
#     for j in range(col):    
#     # get data as stroke-3 format data.
#         ax = plt.subplot(gs[i,j])
#         view_stroke3(strk_dat1[c],ax)
#         ax.set_title(str(c+1))
#         c+=1

# plt.show()       

# print(len(train[0][0]))