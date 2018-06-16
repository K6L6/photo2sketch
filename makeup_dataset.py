from svg_parser import svg2xyList, to_stroke3, arr_reduce
import os
import glob
import numpy as np
import ipdb
import argparse

def convert_svgs(svg_dir, npz_filename, max_length=174, num_train=100, num_valid=1):
    """ convert all the svg files in svg_dir. """
    print('convert svg files in ', svg_dir)

    # get filenames
    # svg_filenames = glob.glob(svg_dir + '/*.svg')
    svg_filenames = []
    _svg = [
    '180-1.svg',
    '208-4.svg',
    '318-2.svg',
    '374-4.svg',
    '567-5.svg',
    '748-9.svg',
    '793-4.svg',
    '894-2.svg',
    '1258-1.svg',
    '1507-9.svg',
    '1565-5.svg',
    '1627-7.svg',
    '1635-2.svg',
    '1872-8.svg',
    '2025-1.svg',
    '2379-5.svg',
    '2580-2.svg',
    '2733-1.svg',
    '2756-9.svg',
    '2769-6.svg',
    '3311-2.svg',
    '3854-6.svg',
    '4117-9.svg',
    '4233-6.svg',
    '4357-4.svg',
    '4452-5.svg',
    '4545-5.svg',
    '4618-6.svg',
    '4634-3.svg',
    '4766-2.svg',
    '5062-5.svg',
    '5779-4.svg',
    '5781-5.svg',
    '5967-4.svg',
    '6178-2.svg',
    '6390-5.svg',
    '6410-9.svg',
    '6976-2.svg',
    '7028-5.svg',
    '7384-4.svg',
    '7601-7.svg',
    '7694-5.svg',
    '7848-9.svg',
    '7905-3.svg',
    '7935-4.svg',
    '8766-3.svg',
    '8793-4.svg',
    '8886-4.svg',
    '8961-2.svg',
    '9124-9.svg',
    '9312-3.svg',
    '10213-7.svg',
    '10233-3.svg',
    '10236-2.svg',
    '10347-5.svg',
    '10564-5.svg',
    '10579-1.svg',
    '10648-4.svg',
    '10987-3.svg',
    '11247-2.svg',
    '11310-7.svg',
    '11686-5.svg',
    '11734-5.svg',
    '11769-7.svg',
    '11895-3.svg',
    '12505-5.svg',
    '12672-2.svg',
    '12814-10.svg',
    '12857-6.svg',
    '12894-2.svg',
    '12978-6.svg',
    '13023-9.svg',
    '13068-6.svg',
    '13385-6.svg',
    '13869-4.svg',
    '13900-1.svg',
    '13922-8.svg',
    '14074-1.svg',
    '14112-1.svg',
    '14195-7.svg',
    '14474-6.svg',
    '14495-2.svg',
    '14541-1.svg',
    '15585-4.svg',
    '15624-2.svg',
    '15919-1.svg',
    '16219-7.svg',
    '16309-4.svg',
    '16335-5.svg',
    '17160-2.svg',
    '17425-8.svg',
    '17579-5.svg',
    '17887-8.svg',
    '17921-6.svg',
    '18305-2.svg',
    '18533-3.svg',
    '18695-1.svg',
    '18877-5.svg',
    '19895-3.svg',
    '23229-4.svg'
    ]

    for i in range(len(_svg)):
        svg_filenames.append(svg_dir+'n01621127_'+_svg[i])
    # ipdb.set_trace()

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
        valid=strokes[:num_train],
        # test=strokes[num_train:],
        test=strokes[:num_train],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for converting svg files to npz dataset file for sketch-rnn.'
    )
    parser.add_argument('svg_dir', help='directory that has many svg files.')
    parser.add_argument('npz_filename', help='filename of converted npz filename')
    args = parser.parse_args()

    convert_svgs(args.svg_dir, args.npz_filename)