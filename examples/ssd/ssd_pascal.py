from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

def AddExtraLayers(net, use_batchnorm=True):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    # Add additional convolutional layers.
    name = "conv6_1"
    net[name] = L.Convolution(net.fc7, num_output=512, kernel_size=1, **kwargs)
    if use_batchnorm:
        batchnorm_name = "{}_bn".format(name)
        net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
    net.relu6_1 = L.ReLU(net[name], in_place=True)
    name = "conv6_2"
    net[name] = L.Convolution(net.relu6_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    if use_batchnorm:
        batchnorm_name = "{}_bn".format(name)
        net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
    net.relu6_2 = L.ReLU(net[name], in_place=True)
    name = "conv6_3"
    net[name] = L.Convolution(net.relu6_2, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    if use_batchnorm:
        batchnorm_name = "{}_bn".format(name)
        net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
    net.relu6_3 = L.ReLU(net[name], in_place=True)

    # Add global pooling layer.
    net.pool6 = L.Pooling(net.relu6_3, pool=P.Pooling.AVE, global_pooling=True)

    return net

# configuration for PASCAL VOC
num_classes = 21
save_dir = "examples/VOC0712"
train_data = "examples/VOC0712/VOC0712_trainval_lmdb"
test_data = "examples/VOC0712/VOC0712_test_lmdb"
label_map_file = "data/VOC0712/labelmap_voc.prototxt"
min_dim = 600
max_dim = 1000
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
mbox_source_layers = ['conv4_3', 'conv5_3', 'fc7', 'conv6_3', 'pool6']
min_sizes = [40, 92, 196, 404, 600]
max_sizes = [92, 196, 404, 596, 1000]
aspect_ratios = [[2, 4], [2, 4], [2, 3, 4], [2], [2, 3]]
# mbox_source_layers = ['fc7', 'conv6_3', 'pool6']
# min_sizes = [196, 404, 600]
# max_sizes = [404, 596, 600]
aspect_ratios = [[2, 3, 4], [2, 3], [2]]
use_batchnorm = True
multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.L2,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': 1,
        'num_classes': num_classes,
        'share_location': True,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True}

pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc.caffemodel"
train_net_file = "examples/VOC0712/train.prototxt"
test_net_file = "examples/VOC0712/test.prototxt"
solver_file = "examples/VOC0712/solver.prototxt"
snapshot_prefix = "models/VGGNet/VOC0712/VGG_SSD_PASCAL0712"

# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=1,
        train=True, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead(net, data_layer="data", from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, num_classes=num_classes, share_location=True,
        flip=True, clip=True)

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "VGG_SSD_PASCAL0712_train"', file=f)
    print(net.to_proto(), file=f)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=1,
        train=False, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead(net, data_layer="data", from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, num_classes=num_classes, share_location=True,
        flip=True, clip=True)

with open(test_net_file, 'w') as f:
    print('name: "VGG_SSD_PASCAL0712_test"', file=f)
    print(net.to_proto(), file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        base_lr=0.4,
        lr_policy="step",
        gamma=0.1,
        stepsize=6000,
        display=2,
        average_loss=10,
        max_iter=50000,
        iter_size=8,
        momentum=0.9,
        weight_decay=0.0005,
        snapshot=6000,
        snapshot_prefix=snapshot_prefix,
        solver_mode=P.Solver.GPU,
        device_id=1,
        snapshot_after_train=True,
        solver_type=P.Solver.SGD,
        debug_info=False)

with open(solver_file, 'w') as f:
    print(solver, file=f)

