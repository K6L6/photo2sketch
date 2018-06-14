import numpy as np
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model

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
img = image.load_img('panda.jpg', target_size=(224, 224))
x = image.img_to_array(img)

# x is 3D matrix [224, 224, 3]
# to make a batch, just add new axis.
x = np.expand_dims(x, axis=0)

# if you want to input multiple images at once,
# make 3d matrices and stack them into 4D batch.

# preprocess batch data.
x = preprocess_input(x)

# get the output of intermediate layer.
preds = model_feat_extract.predict(x)

print(type(preds))
print(preds.shape)