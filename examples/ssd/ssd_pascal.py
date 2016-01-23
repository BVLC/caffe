from __future__ import print_function
import caffe
from caffe.model_libs import *

def AddExtraLayers(net):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    # Add additional convolutional layers.
    net.conv6_1 = L.Convolution(net.fc7, num_output=512, kernel_size=1, **kwargs)
    net.relu6_1 = L.ReLU(net.conv6_1, in_place=True)
    net.conv6_2 = L.Convolution(net.relu6_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu6_2 = L.ReLU(net.conv6_2, in_place=True)
    net.conv6_3 = L.Convolution(net.relu6_2, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    net.relu6_3 = L.ReLU(net.conv6_3, in_place=True)

    # Add global pooling layer.
    net.pool6 = L.Pooling(net.relu6_3, pool=P.Pooling.AVE, global_pooling=True)

    return net

# configuration for PASCAL VOC
num_classes = 21
train_data = "example/VOC0712/VOC0712_trainval_lmdb"
test_data = "example/VOC0712/VOC0712_test_lmdb"
min_dim = 600
max_dim = 1000
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
mbox_source_layers = ['conv4_3', 'conv5_3', 'fc7', 'conv6_3', 'pool6']
min_sizes = [40, 92, 196, 404, 600]
max_sizes = [92, 196, 404, 596, 1000]
aspect_ratios = [[3], [3], [2, 3], [2, 3], [2, 3]]
multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.L2,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': 0.06,
        'num_classes': num_classes,
        'share_location': True,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True}

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=2,
        train=True, output_label=True)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net)

CreateMultiBoxHead(net, label_layer="label", from_layers=mbox_source_layers,
        min_sizes=min_sizes, max_sizes=max_sizes, num_classes=num_classes,
        share_location=True, aspect_ratios=aspect_ratios, flip=True, clip=True,
        multibox_loss_param=multibox_loss_param)

with open('examples/VOC0712/train.prototxt', 'w') as f:
    print('name: "VGG_SSD_PASCAL0712_train"', file=f)
    print(net.to_proto(), file=f)


# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=2,
        train=False, output_label=True)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net)

with open('examples/VOC0712/test.prototxt', 'w') as f:
    print('name: "VGG_SSD_PASCAL0712_test"', file=f)
    print(net.to_proto(), file=f)

# net.data = CreateAnnotatedDataLayer(train_data, batch_size=4, train=False)
