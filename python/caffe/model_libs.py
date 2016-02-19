import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output, kernel_size, pad, stride, conv_prefix='', bn_prefix='bn_', scale_prefix='scale_'):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False
        }
    # parameters for scale bias layer after batchnorm.
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=1.00001)],
        'filler': dict(type='constant', value=1),
        'bias_filler': dict(type='constant', value=0.001)
        }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}'.format(conv_prefix, out_layer)
  net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)
  if use_bn:
    bn_name = '{}{}'.format(bn_prefix, out_layer)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True)
    sb_name = '{}{}'.format(scale_prefix, out_layer)
    net[sb_name] = L.Scale(net[conv_name], in_place=True, **sb_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  bn_prefix = 'bn{}_'.format(block_name)
  scale_prefix = 'scale{}_'.format(block_name)

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, conv_prefix=conv_prefix,
        bn_prefix=bn_prefix, scale_prefix=scale_prefix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, conv_prefix=conv_prefix,
      bn_prefix=bn_prefix, scale_prefix=scale_prefix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, conv_prefix=conv_prefix,
      bn_prefix=bn_prefix, scale_prefix=scale_prefix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, conv_prefix=conv_prefix,
      bn_prefix=bn_prefix, scale_prefix=scale_prefix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)


def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='',
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data


def VGGNetBody(net, need_fc=True, fully_conv=False, reduced=False, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    net.conv1_1 = L.Convolution(net.data, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net.pool1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    net.pool2 = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net.pool2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    net.pool3 = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    net.pool4 = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net.pool4, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if fully_conv:
            if reduced:
                net.pool5 = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
                net.fc6 = L.Convolution(net.pool5, num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
            else:
                net.pool5 = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
                net.fc6 = L.Convolution(net.pool5, num_output=4096, pad=3, kernel_size=7, **kwargs)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def ResNet152Body(net, from_layer, use_pool5=True):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        normalizations=[], use_batchnorm=True, min_sizes=[], max_sizes=[], prior_variance = 0.1,
        aspect_ratios=[], share_location=True, flip=True, clip=True, inter_layer_depth=0):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False, fix_scale=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth > 0:
            inter_name = "{}_inter".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True,
                num_output=inter_layer_depth, kernel_size=1, pad=0, stride=1)
            from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = [2, 3]
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        if max_sizes:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        # Create location prediction layer.
        name = "{}_mbox_loc".format(from_layer)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_loc_output, kernel_size=1, pad=0, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create location prediction layer.
        name = "{}_mbox_conf".format(from_layer)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_conf_output, kernel_size=1, pad=0, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        if max_sizes:
            net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
        else:
            net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
        priorbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])

    return mbox_layers
