from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import os
import stat
import subprocess

def AddExtraLayers(net, use_batchnorm=True):
    if use_batchnorm:
        # No bias in conv.
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'weight_filler': dict(type='gaussian', std=0.01)}
        # parameters for scale bias layer after batchnorm.
        sb_kwargs = {
            'param': [dict(lr_mult=1.00001)],
            'filler': dict(type='constant', value=1),
            'bias_term': True,
            'bias_filler': dict(type='constant', value=0.001)
            }
    else:
        # Use xavier filler.
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
        scalebias_name = "{}_sb".format(batchnorm_name)
        net[scalebias_name] = L.Scale(net[batchnorm_name], in_place=True, **sb_kwargs)
    net.relu6_1 = L.ReLU(net[name], in_place=True)
    name = "conv6_2"
    net[name] = L.Convolution(net.relu6_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    if use_batchnorm:
        batchnorm_name = "{}_bn".format(name)
        net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
        scalebias_name = "{}_sb".format(batchnorm_name)
        net[scalebias_name] = L.Scale(net[batchnorm_name], in_place=True, **sb_kwargs)
    net.relu6_2 = L.ReLU(net[name], in_place=True)
    name = "conv6_3"
    net[name] = L.Convolution(net.relu6_2, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    if use_batchnorm:
        batchnorm_name = "{}_bn".format(name)
        net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
        scalebias_name = "{}_sb".format(batchnorm_name)
        net[scalebias_name] = L.Scale(net[batchnorm_name], in_place=True, **sb_kwargs)
    net.relu6_3 = L.ReLU(net[name], in_place=True)

    # Add global pooling layer.
    net.pool6 = L.Pooling(net.relu6_3, pool=P.Pooling.AVE, global_pooling=True)

    return net


### Modify the following parameters accordingly ###
# configuration for PASCAL VOC
caffe_root = '{}/projects/caffe'.format(os.environ['HOME'])
save_dir = "examples/VOC0712"
train_data = "examples/VOC0712/VOC0712_trainval_lmdb"
test_data = "examples/VOC0712/VOC0712_test_lmdb"
label_map_file = "data/VOC0712/labelmap_voc.prototxt"
# Set true if you want to start training right after generating all files.
run_soon = True
# MultiBoxLoss parameters.
num_classes = 21
share_location = True
background_label_id=0
model_name = "VGG_SSD_PASCAL0712"
use_batchnorm = False
if use_batchnorm:
    base_lr = 0.4
else:
    base_lr = 0.001
batch_size = 12
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.L2,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': 1,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'normalize': True,
    }
# parameters for generating priors.
min_dim = 600
max_dim = 600
# mbox_source_layers = ['conv4_3', 'conv5_3', 'fc7', 'conv6_3', 'pool6']
# min_sizes = [40, 92, 196, 404, 600]
# max_sizes = [92, 196, 404, 596, 1000]
# aspect_ratios = [[2, 4], [2, 4], [2, 3, 4], [2], [2, 3]]
mbox_source_layers = ['fc7', 'conv6_3', 'pool6']
min_sizes = [196, 404, min_dim]
max_sizes = [404, 596, max_dim]
aspect_ratios = [[2, 3, 4], [2, 3], [2]]
flip = True
clip = True
# parameters for solver.
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
iter_size = 12 / batch_size
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'gamma': 0.1,
    'stepsize': 3000,
    'iter_size': iter_size,
    'max_iter': 30000,
    'snapshot': 6000,
    'display': 10,
    'average_loss': 10,
    'solver_type': P.Solver.SGD,
    'solver_mode': P.Solver.GPU,
    'device_id': 1,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [4952],
    'test_interval': 1500,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }
# parameters for non maximum suppression at TEST phase
nms_param = {
    'nms_threshold': 0.3,
    'top_k': 400
    }
# parameters for evaluation
overlap_threshold = 0.5

pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc.caffemodel"
train_net_file = "examples/VOC0712/train.prototxt"
test_net_file = "examples/VOC0712/test.prototxt"
solver_file = "examples/VOC0712/solver.prototxt"
snapshot_dir = "models/VGGNet/VOC0712"
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
job_dir = "jobs/VGGNet/VOC0712"
job_file = "{}/{}.sh".format(job_dir, model_name)


### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size,
        train=True, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, num_classes=num_classes,
        share_location=share_location, flip=flip, clip=clip)

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=1,
        train=False, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, num_classes=num_classes,
        share_location=share_location, flip=flip, clip=clip)

net.detection_out = L.DetectionOutput(*mbox_layers, detection_output_param=dict(num_classes=num_classes,
    share_location=share_location, background_label_id=background_label_id,
    nms_param=nms_param), include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    background_label_id=background_label_id, overlap_threshold=overlap_threshold,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write('--weights="{}" \\\n'.format(pretrain_model))
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(solver_param['device_id'], job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
