from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import os
import stat
import subprocess
import sys

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)

    for i in xrange(7, 9):
      from_layer = out_layer
      out_layer = "conv{}_1".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1)

      from_layer = out_layer
      out_layer = "conv{}_2".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    # Add global pooling layer.
    name = net.keys()[-1]
    net.pool6 = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)

    return net


### Modify the following parameters accordingly ###
# Notice: we do evaluation by setting the solver parameters approximately.
# The reason that we do not use ./build/tools/caffe test ... is because it
# only supports testing for classification problem now.
# The directory which contains the caffe code.
caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True

# The size of the images stored in lmdb in the format of
# minsize x maxsize _ resizewidth x resizeheight
size = "0x0_300x300"
# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "examples/VOC0712/VOC0712_trainval_{}_lmdb".format(size)
# The database file for testing data. Created by data/VOC0712/create_data.sh
test_data = "examples/VOC0712/VOC0712_test_{}_lmdb".format(size)

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.4
else:
    base_lr = 0.001

# The job name should be same as the name used in examples/ssd/ssd_pascal.py.
job_name = "SSD_{}_{}".format(size, base_lr)
# The name of the model. Modify it if you want.
model_name = "VGG_VOC0712_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/VOC0712/{}_score".format(job_name)
# Directory which stores the snapshot of trained models.
snapshot_dir = "models/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/VOC0712/{}_score".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/VOCdevkit/results/VOC2007/{}_score/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Find most recent snapshot.
max_iter = 0
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

if max_iter == 0:
  print("Cannot find snapshot in {}".format(snapshot_dir))
  sys.exit()

# Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
name_size_file = "data/VOC0712/test_name_size.txt"
# Stores LabelMapItem.
label_map_file = "data/VOC0712/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 21
share_location = True
background_label_id=0
train_on_diff_gt = False
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': 1.0,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'normalize': False,
    'use_difficult_gt': train_on_diff_gt,
    'do_neg_mining': True,
    'neg_pos_ratio': 1,
    'neg_overlap': 0.5,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# pool6 ==> 1 x 1
mbox_source_layers = ['fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'pool6']
# in percent %
min_ratio = 20
max_ratio = 100
step = 18
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
aspect_ratios = [[2], [2], [2, 3], [2, 3], [2]]
flip = True
clip = True

# Solver parameters.
# Defining which GPUs to use.
device_id = 0
gpus = "0"

# The number does not matter since we do not do training with this script.
batch_size = 1
iter_size = 1

# Which layers to freeze (no backward) during training.
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']

# Evaluate on whole test set.
num_test_image = 4952
test_batch_size = 8
test_iter = num_test_image / test_batch_size

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "fixed",
    'iter_size': iter_size,
    'max_iter': max_iter,
    'snapshot': max_iter,
    'display': 10,
    'average_loss': 10,
    'type': "AdaGrad",
    'solver_mode': P.Solver.GPU,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': False,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': max_iter,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': True,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.3, 'top_k': 400},
    'save_output_param': {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': "{}/{}".format(caffe_root, label_map_file),
        'name_size_file': "{}/{}".format(caffe_root, name_size_file),
        'num_test_image': num_test_image,
        },
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': "{}/{}".format(caffe_root, name_size_file),
    }

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size,
        train=True, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=True, freeze_layers=freeze_layers)

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
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=True)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, num_classes=num_classes,
        share_location=share_location, flip=flip, clip=clip)

if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  conf_name = "mbox_conf"
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
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
  f.write('--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter))
  f.write('--sighup_effect="stop" \\\n')
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}_test{}.log\n'.format(gpus, job_dir, model_name, max_iter))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
