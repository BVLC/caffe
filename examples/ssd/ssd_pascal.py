from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import os
import stat
import subprocess

def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
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
# configuration for PASCAL VOC
caffe_root = '{}/projects/caffe'.format(os.environ['HOME'])
size = "0x0_300x300"
use_batchnorm = False
if use_batchnorm:
    base_lr = 0.4
else:
    base_lr = 0.001
job_name = "SSD_{}_{}".format(size, base_lr)
save_dir = "models/VGGNet/VOC0712/{}".format(job_name)
snapshot_dir = "models/VGGNet/VOC0712/{}".format(job_name)
job_dir = "jobs/VGGNet/VOC0712/{}".format(job_name)
model_name = "VGG_VOC0712_{}".format(job_name)
train_data = "examples/VOC0712/VOC0712_trainval_{}_lmdb".format(size)
test_data = "examples/VOC0712/VOC0712_test_{}_lmdb".format(size)
label_map_file = "data/VOC0712/labelmap_voc.prototxt"
name_size_file = "data/VOC0712/test_name_size.txt"
pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
output_result_dir = "{}/data/VOCdevkit/results/VOC2007/{}/Main".format(os.environ['HOME'], job_name)
device_id = 0
num_gpus = 4
gpus = "0,1,2,3"
batch_size = 32 / num_gpus
iter_size = 32 / batch_size / num_gpus
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from previously saved snapshot.
resume_training = True
# If true, Remove old model files.
remove_old_models = True
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
    'use_difficult_gt': False,
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
# parameters for solver.
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "fixed",
    'iter_size': iter_size,
    'max_iter': 50000,
    'snapshot': 10000,
    'display': 10,
    'average_loss': 10,
    'type': "AdaGrad",
    'solver_mode': P.Solver.GPU,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [495],
    'test_interval': 2000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }
# parameters for non maximum suppression at TEST phase
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
        },
    }
# parameters for evaluation
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': "{}/{}".format(caffe_root, name_size_file),
    }

train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
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
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=1,
        train=False, output_label=True, label_map_file=label_map_file)

VGGNetBody(net, fully_conv=True, reduced=True, freeze_layers=freeze_layers)

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

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Find most recent snapshot.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  f.write('--sighup_effect="stop" \\\n')
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
