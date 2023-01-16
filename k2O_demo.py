import os
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
#import keras2onnx
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import BatchNormalization

onnx_model_name = 'nyu.onnx'

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None, 'BatchNormalization': BatchNormalization}

model = load_model('/content/drive/MyDrive/model.h5')
#model = load_model('C:\\Users\\singhal\\Downloads\\DenseDepth-master\\model_5000\\model_5000.h5', custom_objects=custom_objects, compile=False)
#onnx_model = keras2onnx.convert_keras(model, model.name)
#onnx.save_model(onnx_model, onnx_model_name)

tf.saved_model.save(model, "tmp_model")
