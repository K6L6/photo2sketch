import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as grdspc
from magenta.models.sketch_rnn import utils
from svg_parser import view_stroke3


file_dir='/home/kelvin/OgataLab/magenta/magenta/models/sketch_rnn/sketchy_data/'
file1 = 'sketchrnn_airplane.npz'
file2 = 'sketchy_airplane_ep2.npz'
strk_dat1 = []
strk_dat2 = []

data1 = np.load(file_dir+file1)
data2 = np.load(file_dir+file2)

train1 = data1['train']
for i in range(100):
    strk_dat1.append(train1[i])

train2 = data2['train']
for i in range(100):
    strk_dat2.append(train2[i])

#stroke-3toline
l = []
for i in range(len(strk_dat2)):
    l.append(utils.strokes_to_lines(strk_dat2[i]))
# print(l)
# for i in range(len(strk_dat1)):
#     l.append(utils.strokes_to_lines(strk_dat1[i]))
# print(l)

#2bigstrokes
s5 = []
for i in range(len(strk_dat2)):
    s5.append(utils.to_big_strokes(strk_dat2[i]))
# print(s5)

"""
c=0
row, col = 10, 10
gs = grdspc.GridSpec(row, col)

for i in range(row):
    for j in range(col):    
    # get data as stroke-3 format data.
        ax = plt.subplot(gs[i,j])
        view_npz(strk_dat1[c],ax)
        ax.set_title(str(c+1))
        c+=1
"""
# plt.show()       

# print(len(train[0][0]))