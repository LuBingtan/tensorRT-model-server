import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data

import pycuda.driver as cuda
import pycuda.autoinit
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser

def addWatch(run_options,node_name,debug_urls):
    watch_opts = run_options.debug_options.debug_tensor_watch_opts
    run_options.debug_options.global_step = 1
    watch = watch_opts.add()
    watch.node_name=node_name
    watch.output_slot=0
    debug_ops="DebugIdentity"
    if isinstance(debug_ops, str):
        debug_ops = [debug_ops]
    watch.debug_ops.extend(debug_ops)
    if debug_urls:
        if isinstance(debug_urls, str):
            debug_urls = [debug_urls]
        watch.debug_urls.extend(debug_urls)

logdir='./'
output_graph_path = logdir+'model.pb'
MNIST_DATASETS = input_data.read_data_sets(
    '/tmp/tensorflow/mnist/input_data')
with tf.Session() as sess:
    # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with tf.gfile.GFile(output_graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        output_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(output_graph_def, name='')
    inputs = sess.graph.get_tensor_by_name("Placeholder:0")
    print inputs
    outputs = sess.graph.get_tensor_by_name("fc2/Relu:0")
    print outputs
    img, label = MNIST_DATASETS.test.next_batch(1)
    img = img[0].reshape((1,28,28,1))
    print type(img)
    rst = sess.run(outputs, feed_dict={inputs:img})
    print rst
    graphdef = tf.get_default_graph().as_graph_def()
    tf_model = tf.graph_util.remove_training_nodes(graphdef)
    uff.from_tensorflow(tf_model, ["fc2/Relu"], output_filename="./test.uff")
    uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"])
    print type(uff_model)
    print len(uff_model)
