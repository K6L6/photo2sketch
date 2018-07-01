import numpy as np
import glob
import ipdb
import csv
from os import listdir
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model

d_dir = "/home/kelvin/OgataLab/sketch-wmultiple-tags/rendered_256x256/256x256/photo/tx_000000000000/owl/"

test_dir = "./test_p"
# define pre-trained model.
# for the other models, see https://keras.io/ja/applications/ .
model = MobileNetV2()

# print model structure.
print(model.summary())

# specify the output layer by name (string)
layer_name = "block_15_add"

# initialize a part of model by reusing the full model.
model_feat_extract = Model(
    inputs=model.input,
    outputs=model.get_layer(layer_name).output)

# load a jpg file, and convert to numpy.array.
# replace 'panda.jpg' to the name of your image file.

# x is 3D matrix [224, 224, 3]
# to make a batch, just add new axis.

# if you want to input multiple images at once,
# make 3d matrices and stack them into 4D batch.

# preprocess batch data.
'''Single image'''

# img = image.load_img('test.jpeg',target_size=(224,224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

'''Multi image'''
# img_arr = glob.glob(d_dir+'/*.jpg')
# img_arr = sorted(img_arr)
img_n = list(range(1,101))
img_arr = []
for i in range(len(img_n)):
    img_arr.append(d_dir+str(img_n[i])+'.jpg')

# ipdb.set_trace()
x_list = []
for i in range(len(img_arr)):
    x_list.append(image.load_img(img_arr[i],target_size=(224,224)))

x=[]
for j in range(len(x_list)):
    x.append(image.img_to_array(x_list[j]))

for k in range(len(x)):
    x[k] = image.img_to_array(x[k])
x = np.asarray(x)

# get the output of intermediate layer.
preds = model_feat_extract.predict(x)

print(type(preds))
print(preds.shape)
predz=[]
for i in range(len(preds)):
    predz.append(np.ndarray.flatten(preds[i]))
predz = np.asarray(predz)

with open("photo_z_tt.csv","w+") as _csv:
    csvw = csv.writer(_csv, delimiter=',')
    csvw.writerows(predz)