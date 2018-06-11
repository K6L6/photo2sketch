import tensorflow as tf
import numpy as np
import glob
import ipdb
import imageio
from scipy import misc

# batch_size=-1
# inp = tf.reshape(features["x"],[batch_size,256,256,1])

jpg_dir = "/home/kelvin/OgataLab/resnet/rendered_256x256/256x256/photo/tx_000000000000/owl"
# f = misc.ndimage.imread()
jpg_filenames = glob.glob(jpg_dir + '/*.jpg')
if len(jpg_filenames) == 0:
    raise RuntimeError("no svg file in {}".format(jpg_dir))

def conv_jpg(f):
    return imageio.imread(f)

invalid_files = []
images = []
    
for i, f in enumerate(jpg_filenames):
    try:
        data = conv_jpg(f)
        images.append(data)
    except Exception as exp:
        print(f, exp)
        invalid_files.append(f)

data=data.flatten()
np.savetxt("owl.csv",data,delimiter=",")