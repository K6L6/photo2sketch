from svg.path import parse_path
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier
from xml.dom import minidom
# from magenta.models.sketch_rnn import utils
from rdp import rdp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import numpy as np
import os
import re
import sys
import ipdb

folder_p = '/home/kelvin/OgataLab/parse_svg/parse_svg/Sketchy_data_valid/airplane/'
dest_p = '/home/kelvin/OgataLab/magenta/magenta/models/sketch_rnn/sketchy_data/'
# f_name = 'n02139199_15837-7.svg'


def adjust_axes(axis):
    """ equalize the scales of x-axis and y-axis """
    x_lim = axis.get_xlim()
    y_lim = axis.get_ylim()
    new_lim = (min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1]))
    axis.set_xlim(new_lim)
    axis.set_ylim(new_lim)
    axis.set_aspect('equal')


def rsvg_in_folderxy(f_path, no_of_files):
    '''reads all the svg files in from folder path, equal to the number of files specified. Then all path data in the svg files are converted into stroke-3 data and stored in a list'''
    c=0
    svg_data = []

    for file in os.listdir(f_path):
        
        if file.endswith(".svg"):
            # data = to_stroke3(svg2xyList(f_path + file))
            data = svg2xyList(f_path + file)
            if c<no_of_files:
                svg_data.append(data)
                c+=1
            else:
                break
    return svg_data

def rsvg_in_folders3(f_path, no_of_files):
    '''reads all the svg files in from folder path, equal to the number of files specified. Then all path data in the svg files are converted into stroke-3 data and stored in a list'''
    c=0
    svg_data = []

    for file in os.listdir(f_path):
        
        if file.endswith(".svg"):
            data = to_stroke3(svg2xyList(f_path + file))
            # data = arr_reduce(data,100)
            # data = svg2xyList(f_path + file)
            # ipdb.set_trace()
            if c<no_of_files:
                svg_data.append(data)
                c+=1
            else:
                break
    return svg_data

def check_max_seq(f_path):

    seq_len = []
    for file in os.listdir(f_path):
        if file.endswith(".svg"):
            N = to_stroke3(svg2xyList(f_path+file))
            seq_len.append((len(N),file))
    # ipdb.set_trace()
    return sorted(seq_len,key=lambda k:k[0])
    # print("Maximum sequence length is:",max[])

def svg2xyList(file, eps=3.0):
    '''converts path component "d" from svg file into xy-coordinate list'''
    line = []
    strt_pts = []
    ed_pts = []
    strt_pts1 = []
    ed_pts1 = []
    # print(file)
    all_path = minidom.parse(file).getElementsByTagName('path')
    # ipdb.set_trace()
    for i in range(len(all_path)):
        clr = all_path[i].getAttribute('stroke')
        clr1 = all_path[i].getAttribute('STROKE')
        if clr == '#000' or clr1 == '#000':
            d = all_path[i].getAttribute('d')
            P = parse_path(d)
            points = []
            
            if len(P)<1:
                pass
            else:
                for j in range(len(P)):
                    abs_ed = P[len(P)-1].end.real, P[len(P)-1].end.imag

                    if isinstance(P[j], CubicBezier) or isinstance(P[j], Line):
                        strt = P[j].start.real, P[j].start.imag
                        ed = P[j].end.real, P[j].end.imag
                        
                        if j == 0:
                            abs_strt = strt
                            points.append(strt)
                            points.append(ed)
                        else:
                            points.append(ed)
                    else:
                        print("What?! th is "+P[j])

                points = rdp(points, epsilon=eps)
                line.append(points)
        else:
            pass
    return line

def to_stroke3(data, dtype=np.int16, vmax=500):
    '''converts xy-coordinate list into stroke-3 format numpy array'''

    buf = []

    init_d = np.array(data[0][0])
    l_d = np.array((0, 0))

    for d in data:
        # move to (0, 0)
        d_arr = np.array(d) - init_d

        # convert to relative points
        tmp = d_arr[1:, :] - d_arr[:-1, :]
        tmp = np.concatenate((
            np.expand_dims(d_arr[0, :] - l_d, 0),
            tmp
        ))
        
        # probability of the pen status
        p = np.zeros((tmp.shape[0], 1))
        p[-1, 0] = 1.0

        tmp = tmp
        buf.append(np.concatenate((tmp, p), axis=1))

        # for the next line
        l_d = d_arr[-1, :]
    
    buf = np.vstack(buf)
    
    # normalization to avoid lose of data in casting
    buf[:, :2] = buf[:, :2] / np.absolute(buf[:, :2]).max() * abs(vmax)
    buf = buf.astype(dtype)
    return buf


def get_colors(n, cmap='cool'):
    cmap = matplotlib.cm.get_cmap(cmap)
    idx = np.linspace(0.0, 1.0, n)
    return cmap(idx)


def view_stroke3(data, axis):
    '''view stroke-3 data as an image'''

    line = []
    x, y = 0, 0

    colors = get_colors(len(data))

    for point, color in zip(data, colors):
        _x, _y, v = point
        x += _x
        y += _y
        line.append([x, y])
        if v != 0:
            line = np.array(line)
            axis.plot(line[:,0],line[:,1]*-1.0, color=color)
            line = []
    adjust_axes(axis)


def svg_to_npz_ex(f_path, t, v, tst, max_seq):
    '''converts a number of svg files into npz format. npz file will contain 3 sections, train, validation and test. test, validation, and test parameters should be input as integers, and total shouldn't be greater than existing files.'''
    category = re.search('valid/(.+?)/', f_path).group(1)
    train_data = []
    validation_data = []
    test_data = []
    data1 = []
    c=0    
    # print(category)
    for file in os.listdir(f_path):

        if file.endswith(".svg"):
            # print(file)
            data0 = svg2xyList(f_path + file)
            try:
                check = to_stroke3(data0)  # 2d matrix of stroke-3 format pic.
            except:
                print('This freakin file-->'+file)
            
            seq_len = len(check)
            if seq_len>max_seq:
                pass
            else:
                data1 = exp_w_order(data0)
                # ipdb.set_trace()
                for x in range(len(data1)):
                    data2 = to_stroke3(data1[x])
                    if c<t:
                        train_data.append(data2)
                        c += 1
                    elif c >=t and c<(t+v):
                        validation_data.append(data2)
                        c += 1
                    elif c>=(t+v) and c<(t+v+tst):
                        test_data.append(data2)
                        c += 1
                    else:
                        break
    # ipdb.set_trace()
    filename = os.path.join(dest_p, 'sketchy_'+category+'_exep2.npz')
    np.savez_compressed(filename, train=train_data, valid=validation_data, test=test_data)
    print('conversion succeeded.')

def svg_to_npz(f_path, t, v, tst, max_seq):
    '''converts a number of svg files into npz format. npz file will contain 3 sections, train, validation and test. test, validation, and test parameters should be input as integers, and total shouldn't be greater than existing files.'''
    category = re.search('valid/(.+?)/', f_path).group(1)
    train_data = []
    validation_data = []
    test_data = []
    c=0    
    # print(category)
    for file in os.listdir(f_path):

        if file.endswith(".svg"):
            # print(file)
            data0 = svg2xyList(f_path + file)
            try:
                data1 = to_stroke3(data0)  # 2d matrix of stroke-3 format pic.
            except:
                print('This freakin file-->'+file)
            data1 = arr_reduce(data1,100)
            seq_len = len(data1)
            if seq_len>max_seq:
                pass
            else:
                if c<t:
                    train_data.append(data1)
                    c += 1
                elif c >=t and c<(t+v):
                    validation_data.append(data1)
                    c += 1
                elif c>=(t+v) and c<(t+v+tst):
                    test_data.append(data1)
                    c += 1
                else:
                    break
    # ipdb.set_trace()
    filename = os.path.join(dest_p, 'sketchy_'+category+'_ep3.npz')
    np.savez_compressed(filename, train=train_data, valid=validation_data, test=test_data)
    print('conversion succeeded.')

def view_xylist(data, axis):
    line = []
    # data = np.concatenate(np.array(data)).astype(np.float16)
    for point in data:
        # ipdb.set_trace()        
        line = np.array(point)
        axis.plot(line[:,0],line[:,1]*-1.0)
        line = []
    axis.set_aspect('equal')

def svg_mix(data):
    '''recursively changes the starting stroke of the sketch sequence'''
    fin_lis = []
    for i in range(len(data)):        
        tmp_lis = data
        tmp = [tmp_lis[x] for x in range(1,len(tmp_lis))]
        tmp.append(tmp_lis[0])
        data=tmp
        fin_lis.append(tmp)
        tmp=[]    
    return fin_lis

def svg_reverse(data):
    '''reverses direction of stroke'''
    fin_lis = []
    for i in reversed(range(len(data))):
        fin_lis.append(data[i])    
    return fin_lis

def exp_w_order(data):
    '''expands xy-coordinate data. Data produced from svg2xyList.'''
    data1 = svg_mix(data)
    reverse_lis = []
    for i in range(len(data)):
        inter =svg_reverse(data[i])
        reverse_lis.append(inter)
    data2 = svg_mix(reverse_lis)
    data_ex = data1+data2
    return data_ex    

def arr_reduce(data, lim):
    '''used to reduce length of sequence. 'lim' refers to preferred sequence length.'''
    if len(data)>lim:
        a = [x for x in range(lim,len(data))]
        b = np.delete(data,a,0)
    else:
        b = data
    return b

# data0 = rsvg_in_folderxy(folder_p,500,250)

# data1 = exp_w_order(data0)

# svg_to_npz_ex(folder_p, 15000, 500, 500, 250)
# svg_to_npz(folder_p, 400, 50, 50, 100) #dataset less than 500

# svg2xyList(folder_p+f_name)
# svg_list = rsvg_in_folder(folder_p, 36)
# # ipdb.set_trace()
# c = 0
# row, col = 5, 4
# gs = grdspc.GridSpec(row, col)

# for i in range(row):
#     for j in range(col):    
#         ax = plt.subplot(gs[i,j])
#         view_xylist(data1[c], ax)
#         ax.set_title(str(c+1))
#         c+=1

# for i in range(row):
#     for j in range(col):    
#         ax = plt.subplot(gs[i,j])
#         view_stroke3(svg_list[c], ax)
#         ax.set_title(str(c+1))
#         c+=1

# plt.show()                
# print(P)