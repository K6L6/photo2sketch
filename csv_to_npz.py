import numpy as np
import argparse
import csv

csv_dir = "/home/kelvin/OgataLab/sketch-wmultiple-tags/"
sketch_csv = "owl_z_tt.csv"
photo_csv = "photo_z_tt.csv"

def convert_csv(npz_filename):
    def csv_parse(f):
        data = []
        with open(f,'rb') as cf:
            rd = csv.reader(cf, delimiter = ',')
            for row in rd:
                data.append(map(float,row))
        return np.asarray(data)

    sketch_vec = csv_parse(sketch_csv)
    photo_vec = csv_parse(photo_csv)

    np.savez_compressed(
            npz_filename,
            train_photo = sketch_vec[:-5],
            train_sketch = photo_vec[:-5],
            test_photo = sketch_vec[-5:],
            test_sketch = photo_vec[-5:],
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for converting csv files to npz dataset file for linear regression.'
    )
    parser.add_argument('npz_filename', help='filename of converted npz filename')
    args = parser.parse_args()

    convert_csv(args.npz_filename)