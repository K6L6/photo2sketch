import numpy as np
import argparse
import csv
import ipdb

csv_dir = "/home/kelvin/OgataLab/sketch-wmultiple-tags/dataset/"
sketch_csv = "pig_z_tt.csv"
photo_csv = "pig_photo_z.csv"

npz_dir = "./dataset/"
do = "convert"

def convert_csv(npz_filename):
    def csv_parse(f):
        data = []
        with open(f,'rb') as cf:
            rd = csv.reader(cf, delimiter = ',')
            for row in rd:
                data.append(map(float,row))
        return np.asarray(data)

    sketch_vec = csv_parse(csv_dir+sketch_csv)
    photo_vec = csv_parse(csv_dir+photo_csv)
    
    # photo data
    test_set_p = photo_vec[0:4]
    train_set_p = np.delete(photo_vec, slice(0,4), 0)
    
    # sketch data
    test_set_s = sketch_vec[0:4]
    train_set_s = np.delete(sketch_vec, slice(0,4), 0)
    
    ipdb.set_trace()
    np.savez_compressed(
            npz_filename,
            train_photo = train_set_p,
            train_sketch = train_set_s,
            test_photo = test_set_p,
            test_sketch = test_set_s,
        )

def combine_npz(npz_filename):
    pig = np.load(npz_dir+"pig.npz")
    elephant = np.load(npz_dir+"elephant.npz")

    # train
    train_input = np.append(elephant['train_photo'],pig['train_photo'],0)
    train_target = np.append(elephant['train_sketch'],pig['train_sketch'],0)

    # test
    test_input = np.append(elephant['test_photo'],pig['test_photo'],0)
    test_target = np.append(elephant['test_sketch'],pig['test_sketch'],0)

    np.savez_compressed(
            npz_filename,
            train_photo = train_input,
            train_sketch = train_target,
            test_photo = test_input,
            test_sketch = test_target,
    ) 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for converting csv files to npz dataset file for linear regression.'
    )
    parser.add_argument('npz_filename', help='filename of converted npz filename')
    args = parser.parse_args()
    if do=="convert":
        convert_csv(args.npz_filename)
    elif do=="combine":
        combine_npz(args.npz_filename)