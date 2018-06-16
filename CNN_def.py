import tensorflow as tf
import numpy as np
import resnet_model
import glob
import ipdb
import imageio
import csv
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
        data = data.flatten()
        images.append(data)
    except Exception as exp:
        print(f, exp)
        invalid_files.append(f)
ipdb.set_trace()
with open("owl_im.csv","w+") as my_csv:
  csvw = csv.writer(my_csv,delimiter=',')
  csvw.writerows(images)
# np.savetxt("owl.csv",data,delimiter=",")
# inp = tf.reshape(images,[-1,256,256,1])
# ipdb.set_trace()

def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
    
    dataset = inp
    
    return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
    )

def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(_HEIGHT, _WIDTH, _NUM_CHANNELS)

class P2SModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, resnet_version=resnet_model.DEFAULT_VERSION, dtype=resnet_model.DEFAULT_DTYPE):
    """
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(P2SModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )
