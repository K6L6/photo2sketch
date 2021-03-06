from svg.path import parse_path
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier
from collections import Iterable
from xml.dom import minidom
from rdp import rdp
from svg_parser import rsvg_in_folderxy, rsvg_in_folders3
from itertools import groupby
# from magenta.models.sketch_rnn import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import numpy as np
import itertools
import os
import re
import ipdb

# folder_p = '/home/kelvin/Downloads/'
folder_p = '/home/kelvin/OgataLab/parse_svg/parse_svg/Sketchy_data_valid/airplane/'
# file_n = 'n02691156_58-1.svg'
# file1 = folder_p+file_n

def flatten(lis):
    
    for i in lis:
        if isinstance(i, Iterable) and not isinstance(i,basestring):
            for j in flatten(i):
                yield j
        else:
            yield i

def w_wlines(all_path):
    orig = []
    for i in range(len(all_path)):
        d = all_path[i].getAttribute('d')
        P = parse_path(d)
        points = []
        if len(P)<1:
            pass
        else:
            for j in range(len(P)):
                
                if isinstance(P[j], CubicBezier) or isinstance(P[j], Line):
                    strt = P[j].start.real, P[j].start.imag
                    ed = P[j].end.real, P[j].end.imag
                    
                    if j == 0:
                        points.append(strt)
                        points.append(ed)
                    else:
                        points.append(ed)
                else:
                    print("What?! th is "+P[j])
        
            points = rdp(points, epsilon=0.5)
            orig.append(points)
    return orig

def wo_wlines(all_path):
    line = []
    for i in range(len(all_path)):
        clr = all_path[i].getAttribute('stroke')
        if clr == '#000':
            d = all_path[i].getAttribute('d')
            P = parse_path(d)
            points = []
            if len(P)<1:
                pass
            else:
                for j in range(len(P)):
                    
                    if isinstance(P[j], CubicBezier) or isinstance(P[j], Line):
                        strt = P[j].start.real, P[j].start.imag
                        ed = P[j].end.real, P[j].end.imag
                        
                        if j == 0:
                            points.append(strt)
                            points.append(ed)
                        else:
                            points.append(ed)
                    else:
                        print("What?! th is "+P[j])
            
                points = rdp(points, epsilon=0.5)
                line.append(points)
        else:
            pass
    return line

seq_l = []
svg_data = rsvg_in_folders3(folder_p,100)
# l = []
# for i in range(len(svg_data)):
#     l.append(utils.lines_to_strokes(svg_data[i]))

# my_dat = rsvg_in_folders3(folder_p,500)

# ipdb.set_trace()
c=0

# ipdb.set_trace()
'''histogram generation'''
for i in range(len(svg_data)):
    seq_l.append(len(svg_data[i]))
    if len(svg_data[i])<=100:
        c+=1

seq_l.sort()
y_val = [len(list(group)) for key, group in groupby(seq_l)]
seq_l1 = list(set(seq_l))
print('max seq.<=100 is '+str(c))
plt.xlabel("sequence length")
# plt.hist(seq_l)
plt.bar(seq_l1,y_val)    

'''plotting multiple sequences script'''
# ipdb.set_trace()
# line = np.array(line)
# ipdb.set_trace()
# A = w_wlines(all_path)
# B = wo_wlines(all_path)
# ipdb.set_trace()
# ax1 = plt.subplot(131)
# for i in range(len(A)):
#     leen = np.array(A[i])
#     # ipdb.set_trace()
#     ax1.plot(leen[:,0],leen[:,1]*-1.0)

# ax2 = plt.subplot(132)
# for i in range(len(B)):
#     leen = np.array(B[i])
#     # ipdb.set_trace()
#     ax2.plot(leen[:,0],leen[:,1]*-1.0)

# ax3 = plt.subplot(133)

# plt.plot(list(range(len(seq_l))),seq_l)

# ax2 = plt.subplot(133)
# for i in range(len(B)):
#     leen = np.array(B[i])
#     # ipdb.set_trace()
#     ax2.plot(leen[:,0]*-1.0,leen[:,1]*-1.0)

plt.show()