from svg.path import parse_path
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier
from xml.dom import minidom
from rdp import rdp
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import numpy as np
import os
import re
import ipdb

folder_p = '/home/kelvin/OgataLab/parse_svg/parse_svg/Sketchy_data_valid/airplane/'
dest_p = '/home/kelvin/OgataLab/magenta/magenta/models/sketch_rnn/sketchy_data/'

def rsvg_in_folder(f_path, no_of_files):
    '''reads all the svg files in from folder path, equal to the number of files specified. Then all path data in the svg files are converted into stroke-3 data and stored in a list'''
    c=0
    svg_data = []

    for file in os.listdir(f_path):
        
        if file.endswith(".svg"):
            data = to_stroke3(svg2xyList(f_path + file))
            # data = svg2xyList(f_path + file)

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
            N = to_stroke3(svg2xyList(folder_p+file))
            seq_len.append((len(N),file))
    # ipdb.set_trace()
    return sorted(seq_len,key=lambda k:k[0])
    # print("Maximum sequence length is:",max[])

def svg2xyList(file):
    '''converts path component "d" from svg file into xy-coordinate list'''
    line = []
    strt_pts = []
    ed_pts = []
    strt_pts1 = []
    ed_pts1 = []

    all_path = minidom.parse(file).getElementsByTagName('path')
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
                    # abs_strt = P[0].start.real, P[0].start.imag
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

                # strt_pts1.append(points[0])
                # ed_pts1.append(points[-1])
                points = rdp(points, epsilon=2)
                # strt_pts.append(abs_strt)
                # ed_pts.append(abs_ed)
                # points[0] = abs_strt
                # points[-1] = abs_ed
                line.append(points)
                # ipdb.set_trace()
                # print(len(points))
                # print(len(line))
        else:
            pass
    # print("start points: "+str(strt_pts1 == strt_pts))
    # print("end points: "+str(ed_pts1 == ed_pts))
    return line

def to_stroke3(data):
    '''converts xy-coordinate list into stroke-3 format numpy array'''
    buf = []
    
    for i in range(len(data)):
        pt = []
        
        for j in range(len(data[i])):
            ini_p = [data[i][j][0], data[i][j][1], 0]
            rel_p = [data[i][j][0]-data[i][j-1][0], data[i][j][1]-data[i][j-1][1], 0]
            nl_p = [data[i][j][0]-data[i-1][-1][0], data[i][j][1]-data[i-1][-1][1], 0]
            
            # if j == 0:
            #     pt.append(ini_p)
            if i==0 and j==0:
                pt.append(ini_p)
            elif i!=0 and j==0:
                pt.append(nl_p)
            else:
                pt.append(rel_p)
        
        pt[-1][-1]=1
        buf.append(pt)
    # ipdb.set_trace()
    return np.concatenate(np.array(buf)).astype(np.float16)

def view_stroke3(data, axis):
    '''view stroke-3 data as an image'''
    line = []
    x, y = 0, 0
    # ipdb.set_trace()
    for point in data:
        # ipdb.set_trace()
        _x, _y, v = point
        
        if v == 0:
            x+=_x
            y+=_y
            line.append([x,y])
        else:
            x+=_x
            y+=_y
            line.append([x,y])
            line = np.array(line)
            axis.plot(line[:,0],line[:,1]*-1.0)
            line = []
            # x,y = 0,0

def svg_to_npz(f_path, t, v, tst, max_seq):
    '''converts a number of svg files into npz format. npz file will contain 3 sections, train, validation and test. test, validation, and test parameters should be input as integers, and total shouldn't be greater than existing files.'''
    category = re.search('valid/(.+?)/', f_path).group(1)
    train_data = []
    validation_data = []
    test_data = []
    # c=0
    # #checks number of files
    # for file in os.listdir(f_path):
    #     if file.endswith(".svg"):
    #         c+=1
    # if (t+v+tst)>c:
    #     print('files specified exceed number of total files in folder')
    c=0    
    # print(category)
    for file in os.listdir(f_path):

        if file.endswith(".svg"):
            # print(file)
            data = svg2xyList(f_path + file)
            # print(file+' converted')
            try:
                data = to_stroke3(data)  # 2d matrix of stroke-3 format pic.
            except:
                print('This freakin file-->'+file)
            seq_len = len(data)
            if seq_len>max_seq:
                pass
            else:
                if c<t:
                    train_data.append(data)
                    c += 1
                elif c >=t and c<(t+v):
                    validation_data.append(data)
                    c += 1
                elif c>=(t+v) and c<(t+v+tst):
                    test_data.append(data)
                    c += 1
                else:
                    break
    # ipdb.set_trace()
    filename = os.path.join(dest_p, 'sketchy_'+category+'_ep2.npz')
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

# check_max_seq(folder_p)
# svg_to_npz(folder_p, 400, 50, 50, 250)

# svg_list = rsvg_in_folder(folder_p, 16)
# ipdb.set_trace()
# c = 0
# row, col = 4, 4
# gs = grdspc.GridSpec(row, col)

# for i in range(row):
#     for j in range(col):    
#         ax = plt.subplot(gs[i,j])
#         view_xylist(svg_list[c], ax)
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