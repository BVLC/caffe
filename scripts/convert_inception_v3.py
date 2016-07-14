from __future__ import print_function
import os.path
import re
import sys
import tarfile
import time
from datetime import datetime

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile
import h5py
import math

os.environ["GLOG_minloglevel"] ="3"
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

paddings = {'VALID': [0, 0], 'SAME': [1, 1]}

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
        'model_dir', '/tmp/imagenet',
        """Path to classify_image_graph_def.pb, """
        """imagenet_synset_to_human_label_map.txt, and """
        """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
        """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
        """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

cur_dir = os.path.dirname(os.path.realpath(__file__))
caffe_root = '{}/../'.format(cur_dir)
labelmap_file = caffe_root + 'data/ILSVRC2016/labelmap_ilsvrc_clsloc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(label):
    num_labels = len(labelmap.item)
    found = False
    for i in xrange(0, num_labels):
        if label == labelmap.item[i].label:
            found = True
            return labelmap.item[i].display_name
    assert found == True

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.device('/cpu:0'):
            _ = tf.import_graph_def(graph_def, name='')

def make_padding(padding_name, conv_shape):
    if padding_name == 'VALID':
        return [0, 0]
    elif padding_name == 'SAME':
        return [int(math.ceil(conv_shape[0]/2)), int(math.ceil(conv_shape[1]/2))]
    else:
        sys.exit('Invalid padding name '+padding_name)

def dump_inputlayer(sess, net, operation='create'):
    if operation == 'create':
        resize = sess.graph.get_tensor_by_name('ResizeBilinear/size:0').eval()
        [height, width] = resize
        sub = sess.graph.get_tensor_by_name('Sub/y:0').eval()
        mean = sub
        if not type(mean) is list:
            mean = [float(mean)]
        else:
            mean = [int(x) for x in mean]
        mul = sess.graph.get_tensor_by_name('Mul/y:0').eval()
        scale = float(mul)
        net['data'] = L.Input(shape=dict(dim=[1, 3, int(height), int(width)]), transform_param=dict(mean_value=mean, scale=scale))

def dump_convbn(sess, net, from_layer, out_layer, operation='create'):
    conv = sess.graph.get_operation_by_name(out_layer + '/Conv2D')

    weights = sess.graph.get_tensor_by_name(out_layer + '/conv2d_params:0').eval()
    padding = make_padding(conv.get_attr('padding'), weights.shape)
    strides = conv.get_attr('strides')

    beta = sess.graph.get_tensor_by_name(out_layer + '/batchnorm/beta:0').eval()
    gamma = sess.graph.get_tensor_by_name(out_layer + '/batchnorm/gamma:0').eval()
    mean = sess.graph.get_tensor_by_name(out_layer + '/batchnorm/moving_mean:0').eval()
    std = sess.graph.get_tensor_by_name(out_layer + '/batchnorm/moving_variance:0').eval()

    # TF weight matrix is of order: height x width x input_channels x output_channels
    # make it to caffe format: output_channels x input_channels x height x width
    weights = np.transpose(weights, (3, 2, 0, 1))

    if operation == 'create':
        assert from_layer in net.keys(), '{} not in net'.format(from_layer)

        [num_output, channels, kernel_h, kernel_w] = weights.shape
        [pad_h, pad_w] = padding
        [stride_h, stride_w] = strides[1:3]
        std_eps = 0.001

        # parameters for convolution layer with batchnorm.
        conv_prefix = ''
        conv_postfix = ''
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
            }
        conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
        if kernel_h != kernel_w:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                    kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                    stride_h=stride_h, stride_w=stride_w, **kwargs)
        else:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                    kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)

        # parameters for batchnorm layer.
        bn_prefix = ''
        bn_postfix = '_bn'
        bn_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
            }
        bn_name = '{}{}{}'.format(bn_prefix, conv_name, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True,
                batch_norm_param=dict(eps=std_eps), **bn_kwargs)

        # parameters for scale bias layer after batchnorm.
        bias_prefix = ''
        bias_postfix = '_bias'
        bias_kwargs = {
            'param': [dict(lr_mult=1, decay_mult=0)],
            'filler': dict(type='constant', value=0.0),
            }
        bias_name = '{}{}{}'.format(bias_prefix, conv_name, bias_postfix)
        net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)

        # relu layer.
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)
    elif operation == 'save':
        conv_prefix = ''
        conv_postfix = ''
        conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
        net.params[conv_name][0].data.flat = weights.flat

        # Copy bn parameters.
        bn_prefix = ''
        bn_postfix = '_bn'
        bn_name = '{}{}{}'.format(bn_prefix, conv_name, bn_postfix)
        net.params[bn_name][0].data.flat = mean
        net.params[bn_name][1].data.flat = std
        net.params[bn_name][2].data.flat = 1.

        # Copy scale parameters.
        bias_prefix = ''
        bias_postfix = '_bias'
        bias_name = '{}{}{}'.format(bias_prefix, conv_name, bias_postfix)
        net.params[bias_name][0].data.flat = beta

def dump_pool(sess, net, from_layer, out_layer, operation='create'):
    pooling = sess.graph.get_operation_by_name(out_layer)
    ismax = pooling.type=='MaxPool' and 1 or 0
    ksize = pooling.get_attr('ksize')
    padding = make_padding(pooling.get_attr('padding'), ksize[1:3])
    strides = pooling.get_attr('strides')

    if operation == 'create':
        if ismax:
            pool = P.Pooling.MAX
        else:
            pool = P.Pooling.AVE
        assert from_layer in net.keys()
        [kernel_h, kernel_w] = ksize[1:3]
        [pad_h, pad_w] = padding
        [stride_h, stride_w] = strides[1:3]
        if kernel_h != kernel_w:
            net[out_layer] = L.Pooling(net[from_layer], pool=pool,
                    kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                    stride_h=stride_h, stride_w=stride_w)
        else:
            net[out_layer] = L.Pooling(net[from_layer], pool=pool,
                    kernel_size=kernel_h, pad=pad_h, stride=stride_h)

def dump_softmax(sess, net, from_layer, out_layer, operation='create'):
    softmax_w = sess.graph.get_tensor_by_name('softmax/weights:0').eval()
    softmax_b = sess.graph.get_tensor_by_name('softmax/biases:0').eval()

    softmax_w = np.transpose(softmax_w, (1, 0))

    if operation == 'create':
        assert from_layer in net.keys()
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }
        [num_output, channels] = softmax_w.shape
        net[out_layer] = L.InnerProduct(net[from_layer], num_output=num_output, **kwargs)
        prob_layer = '{}_prob'.format(out_layer)
        net[prob_layer] = L.Softmax(net[out_layer])
    elif operation == 'save':
        net.params[out_layer][0].data.flat = softmax_w.flat
        net.params[out_layer][1].data.flat = softmax_b

def dump_tower(sess, net, from_layer, tower_name, tower_layers, operation='create'):
    for tower_layer in tower_layers:
        tower_layer = '{}/{}'.format(tower_name, tower_layer)
        if 'pool' in tower_layer:
            dump_pool(sess, net, from_layer, tower_layer, operation)
        else:
            dump_convbn(sess, net, from_layer, tower_layer, operation)
        from_layer = tower_layer

def dump_inception(sess, net, inception_name, tower_names, operation='create', final=True):
    if operation == 'create':
        towers_layers = []
        for tower_name in tower_names:
            tower_name = '{}/{}'.format(inception_name, tower_name)
            assert tower_name in net.keys(), tower_name
            towers_layers.append(net[tower_name])
        if final:
            inception_name = '{}/join'.format(inception_name)
        net[inception_name] = L.Concat(*towers_layers, axis=1)

def run_inference_on_image(image):
    if not gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = gfile.FastGFile(image).read()

    # Creates graph from saved GraphDef.
    create_graph()

    # sess = tf.InteractiveSession(config=tf.ConfigProto(
    #         allow_soft_placement=True))
    sess = tf.InteractiveSession()
    ops = sess.graph.get_operations()
    for op in ops:
        print(op.name)
    # Run the graph until softmax
    # start = datetime.now()
    data_tensor = sess.graph.get_tensor_by_name('Mul:0')
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    data = sess.run(data_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = sess.run(softmax_tensor,
            {'DecodeJpeg/contents:0': image_data})
    # time_len = datetime.now() - start
    # print(time_len.microseconds / 1000)
    # print predictions indices and values
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for p in top_k:
        print(get_labelname(p), predictions[p])
    sess.close()

    deploy_net_file = 'models/inception_v3/inception_v3_deploy.prototxt'
    model_file = 'models/inception_v3/inception_v3.caffemodel'
    net = caffe.Net(deploy_net_file, model_file, caffe.TEST)
    net.blobs['data'].reshape(1, 3, 299, 299)
    data = data.transpose(0, 3, 1, 2)

    net.blobs['data'].data.flat = data.flat
    output = net.forward()
    predictions = output['softmax_prob']
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for p in top_k:
        print(get_labelname(p), predictions[p])

def dump_model(operation='create', redo=False):
    # Creates graph from saved GraphDef.
    create_graph()
    sess = tf.InteractiveSession()

    # Creates caffe model.
    deploy_net_file = 'models/inception_v3/inception_v3_deploy.prototxt'
    model_file = 'models/inception_v3/inception_v3.caffemodel'
    net = []

    if operation == 'create' and (not os.path.exists(deploy_net_file) or redo):
        net = caffe.NetSpec()
    elif operation == 'save' and (not os.path.exists(model_file) or redo):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        net = caffe.Net(deploy_net_file, caffe.TEST)
    else:
        return

    # dump the preprocessing parameters
    dump_inputlayer(sess, net, operation)

    # dump the filters
    dump_convbn(sess, net, 'data', 'conv', operation)
    dump_convbn(sess, net, 'conv', 'conv_1', operation)
    dump_convbn(sess, net, 'conv_1', 'conv_2', operation)
    dump_pool(sess, net,  'conv_2', 'pool', operation)
    dump_convbn(sess, net, 'pool', 'conv_3', operation)
    dump_convbn(sess, net, 'conv_3', 'conv_4', operation)
    dump_pool(sess, net,  'conv_4', 'pool_1', operation)

    # inceptions with 1x1, 3x3, 5x5 convolutions
    from_layer = 'pool_1'
    for inception_id in xrange(0, 3):
        if inception_id == 0:
            out_layer = 'mixed'
        else:
            out_layer = 'mixed_{}'.format(inception_id)
        dump_tower(sess, net, from_layer, out_layer,
                ['conv'], operation)
        dump_tower(sess, net, from_layer, '{}/tower'.format(out_layer),
                ['conv', 'conv_1'], operation)
        dump_tower(sess, net, from_layer, '{}/tower_1'.format(out_layer),
                ['conv', 'conv_1', 'conv_2'], operation)
        dump_tower(sess, net, from_layer, '{}/tower_2'.format(out_layer),
                ['pool', 'conv'], operation)
        dump_inception(sess, net, out_layer,
                ['conv', 'tower/conv_1', 'tower_1/conv_2', 'tower_2/conv'], operation)
        from_layer = '{}/join'.format(out_layer)

    # inceptions with 1x1, 3x3(in sequence) convolutions
    out_layer = 'mixed_3'
    dump_tower(sess, net, from_layer, out_layer,
            ['conv'], operation)
    dump_tower(sess, net, from_layer, '{}/tower'.format(out_layer),
            ['conv', 'conv_1', 'conv_2'], operation)
    dump_tower(sess, net, from_layer, out_layer,
            ['pool'], operation)
    dump_inception(sess, net, out_layer,
            ['conv', 'tower/conv_2', 'pool'], operation)
    from_layer = '{}/join'.format(out_layer)

    # inceptions with 1x1, 7x1, 1x7 convolutions
    for inception_id in xrange(4, 8):
        out_layer = 'mixed_{}'.format(inception_id)
        dump_tower(sess, net, from_layer, out_layer,
                ['conv'], operation)
        dump_tower(sess, net, from_layer, '{}/tower'.format(out_layer),
                ['conv', 'conv_1', 'conv_2'], operation)
        dump_tower(sess, net, from_layer, '{}/tower_1'.format(out_layer),
                ['conv', 'conv_1', 'conv_2', 'conv_3', 'conv_4'], operation)
        dump_tower(sess, net, from_layer, '{}/tower_2'.format(out_layer),
                ['pool', 'conv'], operation)
        dump_inception(sess, net, out_layer,
                ['conv', 'tower/conv_2', 'tower_1/conv_4', 'tower_2/conv'], operation)
        from_layer = '{}/join'.format(out_layer)

    # inceptions with 1x1, 3x3, 1x7, 7x1 filters
    out_layer = 'mixed_8'
    dump_tower(sess, net, from_layer, '{}/tower'.format(out_layer),
            ['conv', 'conv_1'], operation)
    dump_tower(sess, net, from_layer, '{}/tower_1'.format(out_layer),
            ['conv', 'conv_1', 'conv_2', 'conv_3'], operation)
    dump_tower(sess, net, from_layer, out_layer,
            ['pool'], operation)
    dump_inception(sess, net, out_layer,
            ['tower/conv_1', 'tower_1/conv_3', 'pool'], operation)
    from_layer = '{}/join'.format(out_layer)

    for inception_id in xrange(9, 11):
        out_layer = 'mixed_{}'.format(inception_id)
        dump_tower(sess, net, from_layer, out_layer,
                ['conv'], operation)
        dump_tower(sess, net, from_layer, '{}/tower'.format(out_layer),
                ['conv'], operation)
        dump_tower(sess, net, '{}/tower/conv'.format(out_layer),
                '{}/tower/mixed'.format(out_layer), ['conv'], operation)
        dump_tower(sess, net, '{}/tower/conv'.format(out_layer),
                '{}/tower/mixed'.format(out_layer), ['conv_1'], operation)
        dump_inception(sess, net, '{}/tower/mixed'.format(out_layer),
                ['conv', 'conv_1'], operation, False)
        dump_tower(sess, net, from_layer, '{}/tower_1'.format(out_layer),
                ['conv', 'conv_1'], operation)
        dump_tower(sess, net, '{}/tower_1/conv_1'.format(out_layer),
                '{}/tower_1/mixed'.format(out_layer), ['conv'], operation)
        dump_tower(sess, net, '{}/tower_1/conv_1'.format(out_layer),
                '{}/tower_1/mixed'.format(out_layer), ['conv_1'], operation)
        dump_inception(sess, net, '{}/tower_1/mixed'.format(out_layer),
                ['conv', 'conv_1'], operation, False)
        dump_tower(sess, net, from_layer, '{}/tower_2'.format(out_layer),
                ['pool', 'conv'], operation)
        dump_inception(sess, net, out_layer,
                ['conv', 'tower/mixed', 'tower_1/mixed', 'tower_2/conv'], operation)
        from_layer = '{}/join'.format(out_layer)

    dump_pool(sess, net, from_layer, 'pool_3', operation)
    dump_softmax(sess, net, 'pool_3', 'softmax', operation)

    if operation == 'create' and (not os.path.exists(deploy_net_file) or redo):
        model_dir = os.path.dirname(deploy_net_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(deploy_net_file, 'w') as f:
            print('name: "inception_v3_deploy"', file=f)
            print(net.to_proto(), file=f)
    elif operation == 'save' and (not os.path.exists(model_file) or redo):
        net.save(model_file)
    sess.close()

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    modelfilepath = os.path.join(dest_directory, 'classify_image_graph_def.pb')
    if not os.path.exists(modelfilepath):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(_):
    maybe_download_and_extract()
    redo = True
    operations = ['create', 'save']
    for operation in operations:
        dump_model(operation, redo)
    eval = True
    if eval:
        image = (FLAGS.image_file if FLAGS.image_file else
               os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
        run_inference_on_image(image)

if __name__ == '__main__':
    tf.app.run()
