###################
# save onnx model #
###################
import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
x = torch.randn(16, 3, 224, 224, requires_grad=True)
torch.onnx.export(model, x, 'tmp/resnet18.onnx', input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}, opset_version=12)


############################
# convert onnx to tf model #
############################
# https://github.com/onnx/onnx-tensorflow/blob/main/example/onnx_to_tf.py
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('tmp/resnet18.onnx')
tf_model = prepare(onnx_model)
tf_model.export_graph('tmp/resnet_tf')


###############################
# convert tf model to tf lite #
###############################
import tensorflow as tf

tflite_model = tf.lite.TFLiteConverter.from_saved_model('tmp/resnet_tf').convert()
with open('tmp/resnet18.tflite', 'wb') as f:
    f.write(tflite_model)

