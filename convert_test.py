from svg_parser import svg2xyList, to_stroke3, arr_reduce, view_stroke3, view_xylist
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def convert_test(svg_filename):
    """ do tests of svg file conversion """
    print('test with ', svg_filename)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(131)

    data_xy = svg2xyList(svg_filename)
    view_xylist(data_xy, ax1)

    ax2 = fig.add_subplot(132)
    
    data_stroke3 = to_stroke3(data_xy)
    view_stroke3(data_stroke3, ax2)

    ax3 = fig.add_subplot(133)

    data_stroke3_reduced = arr_reduce(data_stroke3, 100)
    view_stroke3(data_stroke3_reduced, ax3)

    vmin = data_stroke3_reduced.min()
    vmax = data_stroke3_reduced.max()
    print('min = ', vmin, ' max = ', vmax)

    plt.show()
        

def check_invalid_svg_files(svg_filenames):
    """ convert all of the svg files and test """
    def convert_svg(f):
        data_xy = svg2xyList(f)
        data_stroke3 = to_stroke3(data_xy)
        data_stroke3_reduced = arr_reduce(data_stroke3, 100)
        return data_stroke3_reduced

    print('num of files = ', len(svg_filenames))

    invalid_files = []
    strokes = []

    d_min, d_max = 0, 0

    for i, f in enumerate(svg_filenames):
        try:
            print('try ... ', f)
            data = convert_svg(f)

            d_min = min(data.min())
            d_max = max(data.max())

        except Exception as exp:
            print(f, exp)
            invalid_files.append(f)

    print(invalid_files)
    print('max = ', d_max)
    print('min = ', d_min)

def compare_with_google_dataset(sketch_filename, google_npz_dataset):
    """ check that the format of converted svg sketch data is same as the google npz dataset. """
    def convert_svg(f):
        data_xy = svg2xyList(f)
        data_stroke3 = to_stroke3(data_xy)
        data_stroke3_reduced = arr_reduce(data_stroke3, 100)
        return data_stroke3_reduced

    # load sketch-rnn's dataset
    dataset = np.load(google_npz_dataset, encoding='latin1')
    dataset = dataset['train']

    # print info of google dataset
    d_min, d_max = 0, 0
    for d in dataset:
        d_min = min(d_min, d.min())
        d_max = max(d_max, d.max())
    print('max = ', d_max)
    print('min = ', d_min)
    
    data_google = dataset[0]

    # plot
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    view_stroke3(convert_svg(sketch_filename), ax1)

    ax2 = fig.add_subplot(122)
    view_stroke3(data_google, ax2)

    plt.show()


def run_tests():
    # the path of directory which has all the sketches from sketchy website.
    # http://sketchy.eye.gatech.edu/
    sketchy_dataset_dir = '/Users/kazuma/data/sketches/'
    print('sketchy_dataset_dir = ', sketchy_dataset_dir)

    # the path of google sketch-rnn's dataset
    # https://github.com/hardmaru/sketch-rnn-datasets/tree/master/aaron_sheep
    google_npz_dataset = 'data/aaron_sheep.npz'

    # we focus on sketches of airplane.
    target_class = 'airplane'

    # get a svg
    svg_filenames = glob.glob(os.path.join(sketchy_dataset_dir, target_class) + '/*.svg')
    if len(svg_filenames) == 0:
        raise RuntimeError("no svg file in {}".format(sketchy_dataset_dir))
    
    convert_test(svg_filenames[0])
    # check_invalid_svg_files(svg_filenames)
    # compare_with_google_dataset(svg_filenames[0], google_npz_dataset)

if __name__ == '__main__':
    print('run tests ...')
    run_tests()