import ipdb
import numpy as np
import matplotlib
from itertools import chain

def get_colors(n, cmap='cool'):
    cmap = matplotlib.cm.get_cmap(cmap)
    idx = np.linspace(0.0, 1.0, n)
    return cmap(idx)

def to_abs(data, factor):
    abs_x, abs_y = 0, 0

    x, y = [], []
    r_x, r_y = [], []

    for i in range(len(data)):
        abs_x += float(data[i, 0]) / factor
        abs_y += float(data[i, 1]) / factor
        lift_pen = data[i, 2]

        if lift_pen == 1:
            r_x.append(x)
            r_y.append(y)
            x, y = [], []
        else:
            x.append(abs_x)
            y.append(abs_y * -1.)

    return r_x, r_y

def get_max_lim(min_x, max_x, min_y, max_y, margin=0.1):

    def my_scale(min_x, max_x, scale):
        diff = abs(max_x - min_x)
        center = min_x + diff / 2.0
        min_x = center - (diff / 2.0 * scale)
        max_x = center + (diff / 2.0 * scale)
        return min_x, max_x

    size_x = abs(max_x - min_x)
    size_y = abs(max_y - min_y)

    if size_x > size_y:
        # x-axis is bigger than y
        scale = size_x / size_y
        min_y, max_y = my_scale(min_y, max_y, scale)
    else:
        # y-axis is bigger than x
        scale = size_y / size_x
        min_x, max_x = my_scale(min_x, max_x, scale)

    min_x, max_x = my_scale(min_x, max_x, margin + 1.0)
    min_y, max_y = my_scale(min_y, max_y, margin + 1.0)
    return min_x, max_x, min_y, max_y

def get_min_max(x):
    # get min and max of list of list
    _x = list(chain.from_iterable(x))
    return min(_x), max(_x)

def plot_stroke(ax, data, factor=0.2, lw=2.0):
    x, y = to_abs(data, factor) 
    if len(x) == 0 or len(y) == 0:
        raise ValueError("nothing will be drawn!")

    num_strokes = len(np.where(data[:, -1] == 1)[0])
    colors = get_colors(num_strokes)
    
    for _x, _y, color in zip(x, y, colors):
        ax.plot(_x, _y, color=color, lw=lw)

    min_x, max_x = get_min_max(x)
    min_y, max_y = get_min_max(y)
    
    min_x, max_x, min_y, max_y = get_max_lim(
            min_x, max_x, min_y, max_y)
    
    ax.set_aspect('equal')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xticks([])
    ax.set_yticks([])


