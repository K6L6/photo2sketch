from svg_parser import svg2xyList, to_stroke3, arr_reduce
import os
import glob
import numpy as np
import ipdb
import argparse

def convert_svgs(svg_dir, npz_filename, max_length=100, num_train=600, num_valid=50):
    """ convert all the svg files in svg_dir. """
    print('convert svg files in ', svg_dir)

    # get filenames
    svg_filenames = glob.glob(svg_dir + '/*.svg')
    if len(svg_filenames) == 0:
        raise RuntimeError("no svg file in {}".format(svg_dir))

    def convert_svg(f):
        data_xy = svg2xyList(f)
        data_stroke3 = to_stroke3(data_xy)
        data_stroke3_reduced = arr_reduce(data_stroke3, max_length)
        return data_stroke3_reduced

    print('num of files = ', len(svg_filenames))

    invalid_files = []
    strokes = []
    
    for i, f in enumerate(svg_filenames):
        try:
            data = convert_svg(f)
            strokes.append(data)
        except Exception as exp:
            print(f, exp)
            invalid_files.append(f)

    print(invalid_files)
    
    # save to npz files
    np.savez_compressed(
        npz_filename,
        train=strokes[:num_train],
        valid=strokes[:num_valid],
        test=strokes[num_train:],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for converting svg files to npz dataset file for sketch-rnn.'
    )
    parser.add_argument('svg_dir', help='directory that has many svg files.')
    parser.add_argument('npz_filename', help='filename of converted npz filename')
    args = parser.parse_args()

    convert_svgs(args.svg_dir, args.npz_filename)