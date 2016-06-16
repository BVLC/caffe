from __future__ import division
from __future__ import print_function
import caffe
import numpy as np


from caffe import layers as L
from caffe import params as P

from pdb import set_trace

caffe.set_mode_cpu()


def vgg_net(mode,batch_size=1):
    #This is not the whole network! missing ReLU ect.

    if mode == "cl":
        pad_init = 1
    elif mode == "sg":
        pad_init = 96
    else:
        raise ValueError

    n = caffe.NetSpec()
    p = 1
    pl = P.Pooling.MAX

    n.data = L.DummyData(shape=[dict(dim=[batch_size, 3, 224, 224])],ntop=1)

    n.conv1_1 = L.Convolution(n.data, kernel_size=3, pad=pad_init, num_output=64)
    n.conv1_2 = L.Convolution(n.conv1_1, kernel_size=3, pad=p, num_output=64)
    n.pool1 = L.Pooling(n.conv1_2,       kernel_size=2, stride=2, pool=pl)

    n.conv2_1 = L.Convolution(n.pool1,   kernel_size=3, pad=p, num_output=128)
    n.conv2_2 = L.Convolution(n.conv2_1, kernel_size=3, pad=p, num_output=128)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=pl)

    n.conv3_1 = L.Convolution(n.pool2,   kernel_size=3, pad=p, num_output=256)
    n.conv3_2 = L.Convolution(n.conv3_1, kernel_size=3, pad=p, num_output=256)
    n.conv3_3 = L.Convolution(n.conv3_2, kernel_size=3, pad=p, num_output=256)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=pl)

    n.conv4_1 = L.Convolution(n.pool3,   kernel_size=3, pad=p, num_output=512)
    n.conv4_2 = L.Convolution(n.conv4_1, kernel_size=3, pad=p, num_output=512)
    n.conv4_3 = L.Convolution(n.conv4_2, kernel_size=3, pad=p, num_output=512)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=pl)

    n.conv5_1 = L.Convolution(n.pool4,   kernel_size=3, pad=p, num_output=512)
    n.conv5_2 = L.Convolution(n.conv5_1, kernel_size=3, pad=p, num_output=512)
    n.conv5_3 = L.Convolution(n.conv5_2, kernel_size=3, pad=p, num_output=512)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=pl)


    if mode == "cl":
        n.fc6 = L.InnerProduct(n.pool5, num_output=4096)
        n.fc7 = L.InnerProduct(n.fc6,   num_output=4096)
    elif mode == "sg":
        n.fc6 = L.Convolution(n.pool5, kernel_size=7, pad=0, num_output=4096)
        n.fc7 = L.Convolution(n.fc6,   kernel_size=1, pad=0, num_output=4096)
    else:
        raise ValueError

    return n

def convert_net(base_weights, voc_weights):
    from tempfile import NamedTemporaryFile as TF

    base_model_py = vgg_net(mode="cl")
    voc_model_py = vgg_net(mode="sg")

    # init
    caffe.set_mode_cpu()

    with TF() as f:
        f.write( str(base_model_py.to_proto()) )
        f.flush()
        base_net = caffe.Net(f.name, base_weights, caffe.TEST)


    with TF() as f:
        f.write( str(voc_model_py.to_proto()) )
        f.flush()
        voc_net = caffe.Net(f.name, caffe.TEST)

    # Source and destination paramteres, these are the same because the layers
    # have the same names in base_net, voc_net
    src_params = ['fc6', 'fc7']  # ignore fc8 because it will be re initialized
    dest_params = ['fc6', 'fc7']

    # First: copy shared layers
    shared_layers = set(base_net.params.keys()) & set(voc_net.params.keys())
    shared_layers -= set(src_params + dest_params)

    for layer in sorted(list(shared_layers)):
        print("Copying shared layer",layer)
        voc_net.params[layer][0].data[...] = base_net.params[layer][0].data
        voc_net.params[layer][1].data[...] = base_net.params[layer][1].data

    # Second: copy over the fully connected layers
    # fc_params = {name: (weights, biases)}
    fc_params = {}


    for pr in src_params:
        fc_params[pr] = (base_net.params[pr][0].data, base_net.params[pr][1].data)

    # conv_params = {name: (weights, biases)}
    conv_params = {}
    for pr in dest_params:
        conv_params[pr] = (voc_net.params[pr][0].data, voc_net.params[pr][1].data)

    for pr, pr_conv in zip(src_params, dest_params):
        print('(source) {} weights are {} dimensional and biases are {} dimensional'\
          .format(pr, fc_params[pr][0].shape, fc_params[pr][1].shape))
        print('(destn.) {} weights are {} dimensional and biases are {} dimensional'\
          .format(pr_conv, conv_params[pr_conv][0].shape, conv_params[pr_conv][1].shape))

        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]


    # Third: inititalize upsampling
    #  interp_layers = [k for k in voc_net.params.keys() if 'up' in k]
    # do net surgery to set the deconvolution weights for bilinear interpolation
    #  interp_surgery(voc_net, interp_layers)

    #Finally: Save resulting network
    voc_net.save(voc_weights)


if __name__ == "__main__":
    convert_net("VGG_ILSVRC_16_layers.caffemodel","vgg16fc.caffemodel")
