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

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, mean_value=[104, 117, 123], mirror=True,
        label_map_file=''):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': dict(mean_value=mean_value, mirror=mirror)
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': dict(mean_value=mean_value)
                }
    if output_label:
        data, label = L.AnnotatedData(name="data", label_map_file=label_map_file,
                data_param=dict(batch_size=batch_size, backend=backend, source=source),
                ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.AnnotatedData(name="data", label_map_file=label_map_file,
                data_param=dict(batch_size=batch_size, backend=backend, source=source),
                ntop=1, **kwargs)
        return data

def VGGNetBody(net, fully_conv=False, reduced=False, freeze_layers=[]):
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

    net.pool5 = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    if fully_conv:
        if reduced:
            net.fc6 = L.Convolution(net.pool5, num_output=1024, pad=1, kernel_size=3, dilation=3, **kwargs)
        else:
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

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        normalizations=[], use_batchnorm=True, min_sizes=[], max_sizes=[],
        aspect_ratios=[], share_location=True, flip=True, clip=True):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

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
        kwargs = {
                'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)}

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

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = [2, 3]
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        num_priors_per_location = 2 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        # Create location prediction layer.
        name = "{}_mbox_loc".format(from_layer)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        net[name] = L.Convolution(net[from_layer], num_output=num_loc_output, kernel_size=1, **kwargs)
        if use_batchnorm:
            batchnorm_name = "{}_bn".format(name)
            net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
            scalebias_name = "{}_sb".format(batchnorm_name)
            net[scalebias_name] = L.Scale(net[batchnorm_name], in_place=True, **sb_kwargs)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create location prediction layer.
        name = "{}_mbox_conf".format(from_layer)
        num_conf_output = num_priors_per_location * num_classes;
        net[name] = L.Convolution(net[from_layer], num_output=num_conf_output, kernel_size=1, **kwargs)
        if use_batchnorm:
            batchnorm_name = "{}_bn".format(name)
            net[batchnorm_name] = L.BatchNorm(net[name], in_place=True)
            scalebias_name = "{}_sb".format(batchnorm_name)
            net[scalebias_name] = L.Scale(net[batchnorm_name], in_place=True, **sb_kwargs)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                aspect_ratio=aspect_ratio, flip=flip, clip=clip)
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
