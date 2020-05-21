import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K

# requirement : tensorflow 1.15
assert tf.__version__ == '1.15.2', 'Tensorflow version Error. You need 1.15.2 version'

assert len(sys.argv) == 2, 'Usage: python get_pretrained_model.py [output pb file path]'

output_path = sys.argv[1]
assert output_path[-3:] == '.pb', 'Extension: output file extention is .pb'

# tensorflow r1.15 clone
os.system('git clone -b r1.15 --single-branch https://github.com/tensorflow/tensorflow.git')

IMG_HEIGHT = 299
IMG_WIDTH = 299

# model load (Imagenet pretrained InceptionV3)
# you can choose another pretrained model  
# reference : https://keras.io/ja/applications/
model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), pooling=None, classes=1000)

# output node name
output_node_name = model.output.op.name
# print('Output node name : '.format(model.output.op.name))

# save frozen model at .ckpt file
saver = tf.train.Saver()
saver.save(K.get_session(), 'frozen_model.ckpt')

# convert frozen model from .ckpt to .pb
os.system('python tensorflow/tensorflow/python/tools/freeze_graph.py --input_meta_graph=frozen_model.ckpt.meta --input_checkpoint=frozen_model.ckpt --output_graph={0}  --output_node_names={1} --input_binary=true'.format(output_path,output_node_name))

os.system('rm -rf frozen_model.ckpt*')
os.system('rm checkpoint')