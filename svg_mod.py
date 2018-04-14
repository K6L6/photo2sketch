from svg.path import parse_path
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier
from xml.dom import minidom
from rdp import rdp
from svg_parser import rsvg_in_folder
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import numpy as np
import os
import re
import ipdb

# folder_p = '/home/kelvin/Downloads/'
folder_p = '/home/kelvin/OgataLab/parse_svg/parse_svg/Sketchy_data_valid/airplane/'
# file_n = 'n02691156_58-1.svg'
# file1 = folder_p+file_n

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
svg_data = rsvg_in_folder(folder_p,500)

# ipdb.set_trace()
for i in range(len(svg_data)):
    seq_l.append(len(svg_data[i]))

seq_l.sort()
plt.hist(seq_l)    
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