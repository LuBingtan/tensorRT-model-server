from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
from PIL import Image
import time #import system tools
import os

import uff
import tensorrt as trt
from tensorrt.parsers import uffparser


with open("test.uff", 'rb') as f:
    uff_model = f.read()
print(len(uff_model))
#
MNIST_DATASETS = input_data.read_data_sets(
    '/tmp/tensorflow/mnist/input_data')
#
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (1,28,28),0)
parser.register_output("fc2/Relu")
engine = trt.utils.uff_to_trt_engine(G_LOGGER, 
    uff_model, 
    parser, 
    1, 
    1 << 20)
#host_mem = parser.hidden_plugin_memory()
parser.destroy()


''' inference begin
'''
# input data
img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
#convert input data to Float32
img = img.astype(np.float32)
label = label[0]
# runtime context
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()
#
output = np.empty(10, dtype = np.float32)
#alocate device memory
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
#
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
for i in range(1000):
    print(i)
    # input data
    img, label = MNIST_DATASETS.test.next_batch(1)
    img = img[0]
    #convert input data to Float32
    img = img.astype(np.float32)
    label = label[0]
    #transfer input data to device
    cuda.memcpy_htod_async(d_input, img, stream)
    #execute model
    context.enqueue(1, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #syncronize threads
    stream.synchronize()
    #
    print("Test Case: " + str(label))
    print ("Prediction: " + str(np.argmax(output)))
    ''' inference end
    '''
#
trt.utils.write_engine_to_file("./tf_mnist.engine", engine.serialize())
#
new_engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")
#
context.destroy()
engine.destroy()
new_engine.destroy()
runtime.destroy()