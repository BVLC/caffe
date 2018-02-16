import copy, math

# Import pycaffe
from .net_spec import layers, params, NetSpec, to_proto

from collections import OrderedDict, Counter, Iterable

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import six


class MetaLayers(object):
    def __getattr__(self, name):
        def metalayer_fn(*args, **kwargs):
            fn = None
            netconf = NetConf()
            netconf.parse(kwargs)
            if (name == 'UNet'):
                unetconf = UNetConf()
                unetconf.parse(kwargs)
                fn = implement_usknet(args[0], netconf, unetconf)
            elif (name == 'SKNet'):
                sknetconf = SKNetConf()
                sknetconf.parse(kwargs)
                fn = implement_sknet(args[0], netconf, sknetconf)
            elif (name == 'USKNet'):
                unetconf = UNetConf()
                unetconf.parse(kwargs)
                fn = implement_usknet(args[0], netconf, unetconf)
            return fn
        return metalayer_fn
    
class SKNetConf:
    # SK-Net convolution steps (may change if necessary)
    conv = [[8],[6],[4]]
    pool = [[2],[2],[2]]
    activation = []
    # Feature map increase rule
    fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 1.5))
    # Number of 1x1 (IP) Convolution steps
    ip_depth = 2
    # Feature map increase rule from SK-Convolution to IP
    fmap_bridge_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 4))
    # Feature map decrease rule within IP
    fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 2.5))
    # Network padding
    padding = [44]
    # Hybrid dimensions expressing SW behavior inside SK networks
    hybrid_dimensions = []
    
    def parse(self, params):
        if ('conv' in params):
            self.conv = params['conv']
        if ('pool' in params):
            self.pool = params['pool']
        if ('fmap_inc_rule' in params):
            self.fmap_inc_rule = params['fmap_inc_rule']
        if ('fmap_dec_rule' in params):
            self.fmap_dec_rule = params['fmap_dec_rule']
        if ('ip_depth' in params):
            self.ip_depth = params['ip_depth']
        if ('fmap_bridge_rule' in params):
            self.fmap_bridge_rule = params['fmap_bridge_rule']
        if ('padding' in params):
            self.padding = params['padding']
        if ('activation' in params):
            self.activation = params['activation']
        if ('hybrid_dimensions' in params):
            self.hybrid_dimensions = params['hybrid_dimensions']

    
class UNetConf:
    # Number of U-Net Pooling-Convolution downsampling/upsampling steps
    depth = 3
    # Feature map increase rule (downsampling)
    fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 3))
    # Feature map decrease rule (upsampling)
    fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 3))
    # Skewed U-Net downsampling strategy
    downsampling_strategy = [[2],[2],[2]]
    # U-Net convolution setup (downsampling path)
    conv_down = [[[3],[3]]]
    act_down = []
    # U-Net convolution setup (upsampling path)
    conv_up = [[[3],[3]]]
    act_up = []
    # SK-Net configurations
    sknetconfs = []
    # Upsampling path with deconvolutions instead of convolutions
    use_deconv_uppath = False
    # Use a more stable implementation of upconvolutions
    use_stable_upconv = False
    
    def parse(self, params):
        if ('depth' in params):
            self.depth = params['depth']
        if ('fmap_inc_rule' in params):
            self.fmap_inc_rule = params['fmap_inc_rule']
        if ('fmap_dec_rule' in params):
            self.fmap_dec_rule = params['fmap_dec_rule']
        if ('downsampling_strategy' in params):
            self.downsampling_strategy = params['downsampling_strategy']
        if ('conv_down' in params):
            self.conv_down = params['conv_down']
        if ('act_down' in params):
            self.conv_down = params['act_down']
        if ('conv_up' in params):
            self.conv_up = params['conv_up']
        if ('act_up' in params):
            self.conv_up = params['act_up']
        if ('use_deconv_uppath' in params):
            self.use_deconv_uppath = params['use_deconv_uppath']
        if ('use_stable_upconv' in params):
            self.use_stable_upconv = params['use_stable_upconv']
        if ('sknetconfs' in params):
            for sknetconf_dict in params['sknetconfs']:
                if (sknetconf_dict != None):
                    self.sknetconfs += [SKNetConf()]
                    self.sknetconfs[-1].parse(sknetconf_dict)
                else:
                    self.sknetconfs += [None]
            
class NetConf:
    # Number of feature maps in the start
    fmap_start = 16
    # ReLU negative slope
    relu_slope = 0.005
    # Batch normalization
    use_batchnorm = False
    # Batch normalization moving average fraction
    batchnorm_maf = 0.95
    # Dropout
    dropout = 0.2
    
    def parse(self, params):
        if ('fmap_start' in params):
            self.fmap_start = params['fmap_start']
        if ('relu_slope' in params):
            self.relu_slope = params['relu_slope']
        if ('use_batchnorm' in params):
            self.use_batchnorm = params['use_batchnorm']
        if ('batchnorm_maf' in params):
            self.batchnorm_maf = params['batchnorm_maf']
        if ('dropout' in params):
            self.dropout = params['dropout']


def deconv_act(netconf, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, activation='relu'):
    deconv = L.Deconvolution(bottom, convolution_param=dict(kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant')), param=[dict(lr_mult=1),dict(lr_mult=2)])
    
    # Activation
    if activation == 'relu':
        relu = L.ReLU(deconv, in_place=True, negative_slope=netconf.relu_slope)
        last = relu
    if activation == 'tanh':
        tanh = L.Tanh(deconv, in_place=True)
        last = tanh
    if activation == 'sigmoid':
        sigm = L.Sigmoid(deconv, in_place=True)  
        last = sigm  
    
    if (netconf.dropout > 0):
        drop = L.Dropout(last, in_place=True, dropout_ratio=netconf.dropout)
        last = drop
    
    if (netconf.use_batchnorm == True):
        bnltrain = L.BatchNorm(last, in_place=True, include=[dict(phase=0)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=False, moving_average_fraction=netconf.batchnorm_maf))
        bnltest = L.BatchNorm(last, in_place=True, include=[dict(phase=1)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=True, moving_average_fraction=netconf.batchnorm_maf))
        last = {bnltrain, bnltest}  
    return last

# Convolution block. Order of operations:
# 1. Convolution
# 3. Dropout
# 4. Batchnorm
# 5. ReLU
def conv_act(netconf, bottom, num_output, in_place=True, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, activation='relu'):           
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
    last = conv
           
    # Dropout
    if (netconf.dropout > 0):
        drop = L.Dropout(last, in_place=in_place, dropout_ratio=netconf.dropout)
        last = drop
    
    # Batchnorm
    if (netconf.use_batchnorm == True):
        bnltrain = L.BatchNorm(last, in_place=in_place, include=[dict(phase=0)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=False, moving_average_fraction=netconf.batchnorm_maf))
        bnltest = L.BatchNorm(last, in_place=in_place, include=[dict(phase=1)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=True, moving_average_fraction=netconf.batchnorm_maf))
        last = {bnltrain, bnltest}

    # Activation
    if activation == 'relu':
        relu = L.ReLU(last, in_place=in_place, negative_slope=netconf.relu_slope)
        last = relu
    if activation == 'tanh':
        tanh = L.Tanh(last, in_place=in_place)
        last = tanh
    if activation == 'sigmoid':
        sigm = L.Sigmoid(last, in_place=in_place)  
        last = sigm  
        
    return last
    
def convolution(bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1):      
    return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
    
def max_pool(netconf, bottom, kernel_size=[2], stride=[2], pad=[0], dilation=[1]):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation)
    
def upconv(netconf, bottom, num_output_dec, num_output_conv, kernel_size=[2], stride=[2], stable_mode=False):
    # Stable mode is the more numerically stable pathway
    if stable_mode:
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=kernel_size, stride=stride, pad=[0], group=num_output_dec,
                                 weight_filler=dict(type='constant', value=1), bias_term=False),
                                 param=dict(lr_mult=0, decay_mult=0))
    
        conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], group=1,
                             param=[dict(lr_mult=1),dict(lr_mult=2)],
                             weight_filler=dict(type='msra'),
                             bias_filler=dict(type='constant'))
        return conv
    else:
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_conv, kernel_size=kernel_size, stride=stride, pad=[0], group=1,
                                                            weight_filler=dict(type='msra'), bias_filler=dict(type='constant')),param=[dict(lr_mult=1),dict(lr_mult=2)])
        return deconv
    
def mergecrop(bottom_a, bottom_b, op = 'stack'):
    return L.MergeCrop(bottom_a, bottom_b, forward=[1,1], backward=[1,1], operation=(0 if (op == 'stack') else 1))

    
def implement_sknet(bottom, netconf, sknetconf, return_blobs_only=True):
    blobs = [bottom]
    fmaps = [netconf.fmap_start]
    actidx = 0
    dilation = [1 for i in range(0,len(sknetconf.padding))]
    sw_shape = [minidx(sknetconf.padding, i) + 1 for i in range(0,len(sknetconf.padding))]
    for i in range(0, len(sknetconf.conv)):
        final_ksize = [minidx(sknetconf.conv[i], j) for j in range(0,len(sw_shape))]
        for j in range(0, len(sw_shape)):
            while ((j not in sknetconf.hybrid_dimensions) and (not (sw_shape[j] - (final_ksize[j] - 1)) % minidx(minidx(sknetconf.pool, i), j) == 0 or sw_shape[j] - (final_ksize[j] - 1) < 0)):
                final_ksize[j] += 1
            if j not in sknetconf.hybrid_dimensions:
                # Account for SK-type convolution and pooling analogon in SW network
                sw_shape[j] = (sw_shape[j] - (final_ksize[j] - 1)) / minidx(minidx(sknetconf.pool, i), j)
            else:
                # Hybrid network present where SW = SK in terms of the pooling operation (stride = 1, dilation = 1, pad = 0)
                sw_shape[j] = (sw_shape[j] - (final_ksize[j] - 1)) - (minidx(minidx(sknetconf.pool, i), j) - 1)
        activation = minidx(sknetconf.activation, actidx) if len(sknetconf.activation) > 0 else 'relu'
        actidx = actidx + 1
        conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=final_ksize, dilation=dilation, activation=activation)
        blobs = blobs + [conv]
        pool_kernel_size = minidx(sknetconf.pool, i)
        if (any([x > 1 for x in pool_kernel_size])):
            pool = max_pool(netconf, blobs[-1], kernel_size=pool_kernel_size, stride=[1], dilation=dilation)
            dilation = [(1 if j in sknetconf.hybrid_dimensions else minidx(minidx(sknetconf.pool, i), j) * dilation[j]) for j in range(0, len(dilation))]
            blobs = blobs + [pool]
        if (i < len(sknetconf.conv) - 1):
            fmaps = fmaps + [sknetconf.fmap_inc_rule(fmaps[-1])]

    fmaps = fmaps + [sknetconf.fmap_bridge_rule(fmaps[-1])]
    # 1st IP layer
    activation = minidx(sknetconf.activation, actidx) if len(sknetconf.activation) > 0 else 'relu'
    actidx = actidx + 1
    conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=[max(i,1) for i in sw_shape], dilation=dilation, activation=activation)
    blobs = blobs + [conv]

    # Remaining IP layers
    for i in range(0, sknetconf.ip_depth - 1):
        fmaps = fmaps + [sknetconf.fmap_dec_rule(fmaps[-1])]
        activation = minidx(sknetconf.activation, actidx) if len(sknetconf.activation) > 0 else 'relu'
        actidx = actidx + 1
        conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=[1], activation=activation)
        blobs = blobs + [conv]    
    if return_blobs_only:
        return blobs[-1]
    else:
        return blobs[-1], fmaps[-1]

            

def implement_usknet(bottom, netconf, unetconf, return_blobs_only=True): 
    blobs = [bottom]
    mergecrop_tracker = []
    fmaps = [netconf.fmap_start]
    pad_shape = [[0 for k in range(0, len(unetconf.conv_down[0][0]))] for i in range(0, unetconf.depth + 1)]
    if unetconf.depth > 0:
        # U-Net downsampling; 2*Convolution+Pooling
        for i in range(0, unetconf.depth):
            convolution_config = minidx(unetconf.conv_down, i)
            for j in range(0,len(convolution_config)):
                conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j])
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_down[0][0])):
                    pad_shape[i][k] += (minidx(convolution_config[j], k) - 1)

            mergecrop_tracker += [len(blobs)-1]
            pool = max_pool(netconf, blobs[-1], kernel_size=unetconf.downsampling_strategy[i], stride=unetconf.downsampling_strategy[i])
            blobs = blobs + [pool]
            fmaps = fmaps + [unetconf.fmap_inc_rule(fmaps[-1])]
    
    # If there is no SK-Net component, fill with normal convolutions
    if (unetconf.depth > 0 and (len(unetconf.sknetconfs) - 1 < unetconf.depth or unetconf.sknetconfs[unetconf.depth] == None)):
        convolution_config = minidx(unetconf.conv_down, unetconf.depth)
        for j in range(0,len(convolution_config)):
            # Here we are at the bottom, so the second half of the convolutions already belongs to the up-path
            if (unetconf.use_deconv_uppath and j >= len(convolution_config)/2):
                conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j], pad=[convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))])
                blobs = blobs + [conv]
            else:
                conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j])
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_down[0][0])):
                    pad_shape[unetconf.depth][k] += (minidx(convolution_config[j], k) - 1)
    else:
        netconf_sk = copy.deepcopy(netconf)
        netconf_sk.fmap_start = fmaps[-1]
        sknetconf_sk = copy.deepcopy(unetconf.sknetconfs[unetconf.depth])
        sknetconf_sk.padding = [minidx(sknetconf_sk.padding, i) for i in range(0,len(pad_shape[unetconf.depth]))]
        sk_blob, sk_fmaps = implement_sknet(blobs[-1], netconf_sk, sknetconf_sk, return_blobs_only=False)
        blobs = blobs + [sk_blob]
        fmaps = fmaps + [sk_fmaps]
        for k in range(0, len(unetconf.conv_down[0][0])):
            pad_shape[unetconf.depth][k] += sknetconf_sk.padding[k]
    if unetconf.depth > 0:
        # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
        for i in range(0, unetconf.depth):
            conv = upconv(netconf, blobs[-1], fmaps[-1], unetconf.fmap_dec_rule(fmaps[-1]), kernel_size=unetconf.downsampling_strategy[unetconf.depth - i - 1],
                                       stride=unetconf.downsampling_strategy[unetconf.depth - i - 1], stable_mode=unetconf.use_stable_upconv)
            blobs = blobs + [conv]
            fmaps = fmaps + [unetconf.fmap_dec_rule(fmaps[-1])]
            
            pre_merge_blobs = [blobs[mergecrop_tracker[unetconf.depth - i - 1]]]
            
            # Insert SK-Net in the mergecrop bridge
            if (len(unetconf.sknetconfs) > unetconf.depth - i - 1 and unetconf.sknetconfs[unetconf.depth - i - 1] != None):
                netconf_sk = copy.deepcopy(netconf)
                netconf_sk.fmap_start = fmaps[-1]
                sknetconf_sk = copy.deepcopy(unetconf.sknetconfs[unetconf.depth - i - 1])
                sknetconf_sk.padding = [0 for k in range(0, len(unetconf.conv_down[0][0]))]
                for j in range(unetconf.depth - i, unetconf.depth + 1):
                    for k in range(0, len(unetconf.conv_down[0][0])):
                        sknetconf_sk.padding[k] += pad_shape[j][k] * (j - (unetconf.depth - i - 1)) * 2
                pre_merge_blobs += [implement_sknet(pre_merge_blobs[-1], netconf_sk, sknetconf_sk)]

            # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
            mergec = mergecrop(blobs[-1], pre_merge_blobs[-1])
            blobs = blobs + [mergec]
            
            convolution_config = minidx(unetconf.conv_up, unetconf.depth - i - 1)
            for j in range(0,len(convolution_config)):
                pad =  [convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))] if (unetconf.use_deconv_uppath) else [0]                       
                conv = conv_act(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j], pad=pad)
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_up[0][0])):
                    pad_shape[unetconf.depth - i - 1][k] += (minidx(convolution_config[j], k) - 1)
    # Return the last blob of the network (goes to error objective)
    if return_blobs_only:
        return blobs[-1]
    else:
        return blobs[-1], fmaps[-1]
    
def fix_input_dims(net, source_layers, max_shapes=[], min_shapes=[], shape_coupled=[], phase=None, stage=None, verbose=False):
    """
    This function takes as input:
    net - The network
    source_layers - A list of other inputs to test (note: the nhood input is static and not spatially testable, thus excluded here)
    max_shapes - Maximum spatial dimensions for each source layer
    shape_coupled - A list of spatial dependencies; here [-1, 0] means the Y axis is a free parameter, and the X axis should be identical to the Y axis.
    (The first spatial axis (Z in 3D, Y in 2D and X in 1D) is ALWAYS a free parameter)
    phase - Only include layers of a certain phase for the input fix (0 or 1)
    stage - Only include layers of certain stages for the input fix (list of strings)
    Returns True if successful and False otherwise.
    """

    graph = Graph()
    
    # Resolve the source layer functions   
    for i in range(0, len(source_layers)):
        if (type(source_layers[i]) == net_spec.Top):
            source_layers[i] = source_layers[i].fn

    for name, top in six.iteritems(net.tops):
        if (isinstance(top, Iterable) and len(top) > 0):
            for subtop in top:
                graph.add_element(subtop)
        else:
            graph.add_element(top)
                
    print("Net explicit elements: " + str(len(net.tops)))
    print("Graph nodes: " + str(len(graph.nodes)))
    print("Graph edges: " + str(len(graph.edges)))
    print("Source nodes: " + str(len(graph.get_source_nodes())))
    print("Sink nodes: " + str(len(graph.get_sink_nodes())))

    sources = graph.get_source_nodes()   
    sinks = graph.get_sink_nodes()
    
    test_sources = []
    test_max_shapes = []
    test_min_shapes = []
    
    dims = 0
    
    for i in range(0, len(source_layers)):
        source_layer = source_layers[i]
        for j in range(0, len(sources)):
            source = sources[j]
            if ('dim' in source.fn.params):
                if (source.fn == source_layer):
                    test_sources = test_sources + [source]
                    test_max_shape = source.fn.params['dim']
                    test_min_shape = source.fn.params['dim']
                    if (len(max_shapes) > i):
                        test_max_shape = test_max_shape + max_shapes[i]
                    if (len(min_shapes) > i):
                        test_min_shape = test_min_shape + min_shapes[i]
                    dims = max(dims, len(test_max_shape) - 2)
                    while (len(test_min_shape) < len(test_max_shape)):
                        test_min_shape.append(1)
                    test_max_shapes = test_max_shapes + [test_max_shape]
                    test_min_shapes = test_min_shapes + [test_min_shape]
            elif('input_param' in source.fn.params):
                if (source.fn == source_layer):
                    test_sources = test_sources + [source]
                    test_max_shape = source.fn.params['input_param']['shape']['dim']
                    test_min_shape = source.fn.params['input_param']['shape']['dim']
                    if (len(max_shapes) > i):
                        test_max_shape = test_max_shape + max_shapes[i]
                    if (len(min_shapes) > i):
                        test_min_shape = test_min_shape + min_shapes[i]
                    dims = max(dims, len(test_max_shape) - 2)
                    while (len(test_min_shape) < len(test_max_shape)):
                        test_min_shape.append(1)
                    test_max_shapes = test_max_shapes + [test_max_shape]
                    test_min_shapes = test_min_shapes + [test_min_shape]
            elif('dummy_data_param' in source.fn.params):
                if (source.fn == source_layer):
                    test_sources = test_sources + [source]
                    test_max_shape = source.fn.params['dummy_data_param']['shape']['dim']
                    test_min_shape = source.fn.params['dummy_data_param']['shape']['dim']
                    if (len(max_shapes) > i):
                        test_max_shape = test_max_shape + max_shapes[i]
                    if (len(min_shapes) > i):
                        test_min_shape = test_min_shape + min_shapes[i]
                    dims = max(dims, len(test_max_shape) - 2)
                    while (len(test_min_shape) < len(test_max_shape)):
                        test_min_shape.append(1)
                    test_max_shapes = test_max_shapes + [test_max_shape]
                    test_min_shapes = test_min_shapes + [test_min_shape]
    test_current_shapes = [[] for i in range(0,len(test_sources))]
                
    curr_src_idx = 0
    
    # Test each dimension
    for dim_idx in range(0, dims):
        curr_src_idx = 0
        if (dim_idx > 0 and len(shape_coupled) >= dim_idx and shape_coupled[dim_idx] > -1):
            for src_idx in range(0, len(test_sources)):
                # Check if this source even has one dimension more or not
                if (len(test_current_shapes[src_idx]) < len(test_max_shapes[src_idx])):
                    # Copy the shape from the other dimension
                    test_current_shapes[src_idx] = test_current_shapes[src_idx] + [copy.deepcopy(test_current_shapes[src_idx][shape_coupled[dim_idx] + 2])]
        else:
            # Test each source
            while (True):
                # Initialize the source shape
                if (len(test_current_shapes[curr_src_idx]) == 0):
                    test_current_shapes[curr_src_idx] = [test_max_shapes[curr_src_idx][i] for i in range(0, 2 + dim_idx + 1)]
                elif ((len(test_current_shapes[curr_src_idx]) < 2 + dim_idx + 1) and (len(test_current_shapes[curr_src_idx]) < len(test_max_shapes[curr_src_idx]))):
                    test_current_shapes[curr_src_idx] = test_current_shapes[curr_src_idx] + [test_max_shapes[curr_src_idx][2 + dim_idx]]
                 
                # Forward the values
                error = False
                graph.clear_shapes()
                for idx in range(0, curr_src_idx + 1):
                    graph.propagate_shape_forward(test_sources[idx].fn, idx, test_current_shapes[idx])
                    error = error or graph.has_error(curr_src_idx)

                # Test the shape
                if (verbose or not error):
                    print(test_current_shapes)
                    print("Valid shape: " + str(not error))
                
                if (error and ((len(test_current_shapes[curr_src_idx]) - 2 <= dim_idx) or (test_current_shapes[curr_src_idx][2 + dim_idx] == test_min_shapes[curr_src_idx][2 + dim_idx]))):
                    # Reached minimum shape, reset source and go to previous source
                    if (len(test_current_shapes) - 2 > dim_idx):
                        test_current_shapes[curr_src_idx][2 + dim_idx] = test_max_shapes[curr_src_idx][2 + dim_idx]
                    curr_src_idx = curr_src_idx - 1
                    if (curr_src_idx == -1):
                        # Tested all shapes, found no valid combination of source shapes
                        # Unsuccessful return
                        return False
                # Change the shape
                if (error and test_current_shapes[curr_src_idx][2 + dim_idx] > test_min_shapes[curr_src_idx][2 + dim_idx]):
                    # Error, but still variants left to try, so decrease the dimension
                    test_current_shapes[curr_src_idx][2 + dim_idx] = test_current_shapes[curr_src_idx][2 + dim_idx] - 1
                
                if (not error):
                    if (curr_src_idx == len(test_sources) - 1):
                        # No error at last source element, stop testing for this dimension
                        break
                    else:
                        # Current source has no error, advance to the next source
                        curr_src_idx = (curr_src_idx + 1) % len(test_sources)
                                    
    # Set the shapes
    for src_idx in range(0, len(test_sources)):
        if ('dim' in test_sources[src_idx].fn.params):
            test_sources[src_idx].fn.params['dim'] = test_current_shapes[src_idx]
        elif('input_param' in test_sources[src_idx].fn.params):
            test_sources[src_idx].fn.params['input_param']['shape']['dim'] = test_current_shapes[src_idx]
        elif('dummy_data_param' in test_sources[src_idx].fn.params):
            test_sources[src_idx].fn.params['dummy_data_param']['shape']['dim'] = test_current_shapes[src_idx]
            
    # Successful return
    return True
        

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        
    def reset_error(self):
        for edge in self.edges:
            edge.error = False
        for node in self.nodes:
            node.error = False    
           
    def has_error(self, index):
        error = False
        for edge in self.edges:
            edge.check_shape_errors()
            error = error or edge.error
        for node in self.nodes:
            error = error or node.error
        error = error or self.check_sink_errors(index)
        return error
            
    def clear_shapes(self):
        for edge in self.edges:
            edge.shape = [[]]
            edge.error = False
        for node in self.nodes:
            node.error = False
        
    def get_source_nodes(self):
        source_nodes = []
        for node in self.nodes:
            if (len(node.in_edges) == 0):
                # print(node.fn.type_name)
                source_nodes = source_nodes + [node]
        return source_nodes
            
    def get_sink_nodes(self):
        sink_nodes = []
        for node in self.nodes:
            if (len(node.out_edges) == 0):
                # print(node.fn.type_name)
                sink_nodes = sink_nodes + [node]
        return sink_nodes
    
    def check_sink_errors(self, index):
        error = False
        sink_nodes = self.get_sink_nodes()
        for sink in sink_nodes:
            if (sink.fn.type_name == 'Silence'):
                # Nothing to check, silence terminates blobs of all shapes
                pass
            elif (sink.fn.type_name == 'SoftmaxWithLoss'):
                # Blob 0: Of shape N x C x D x H x W
                # Blob 1: Of shape N x 1 x D x H x W
                prob_shape = []
                label_shape = []
                for idx in range(0, index + 1):
                    for edge_idx in range(0, len(sink.in_edges)):
                        other_shape = sink.in_edges[edge_idx].get_shape(idx)
                        if (edge_idx == 0):
                            if (len(prob_shape) > 0 and len(other_shape) > 0):
                                error = error or not equal_shape(prob_shape, other_shape)
                            elif (len(other_shape) > 0):
                                prob_shape = other_shape
                        elif (edge_idx == 1):
                            if (len(label_shape) > 0 and len(other_shape) > 0):
                                error = error or not equal_shape(label_shape, other_shape)
                            elif (len(other_shape) > 0):
                                label_shape = other_shape
                
                if (len(prob_shape) > 0 and len(label_shape) > 0): 
                    error = error or not (equal_shape(prob_shape[2:], label_shape[2:]))
                    error = error or not (prob_shape[0] == label_shape[0])
                    error = error or not (prob_shape[1] > 1 and label_shape[1] == 1)
                
                # print prob_shape
                # print label_shape
                
            elif (sink.fn.type_name == 'EuclideanLoss'):
                # For euclid, all input shapes should have the same dimension
                # (prediction, target, scale)
                ref_shape = []
                for idx in range(0, index + 1):
                    for i in range(0, len(sink.in_edges)):
                        shape = sink.in_edges[i].get_shape(idx)
                        if (len(ref_shape) == 0):
                            ref_shape = copy.deepcopy(shape)
                        elif (len(shape) > 0):
                            error = error or not equal_shape(ref_shape, shape)
            elif (sink.fn.type_name == 'MalisLoss'):              
                # Blob 0: Of shape N x C x D x H x W
                aff_prob_shape = []
                # Blob 1: Of shape N x C x D x H x W
                aff_shape = []
                # Blob 2: Of shape N x 1 x D x H x W or N x 2 x D x H x W
                components = []
                
                # Load and compare shapes
                for idx in range(0, index + 1):
                    for edge_idx in range(0, len(sink.in_edges)):
                        other_shape = sink.in_edges[edge_idx].get_shape(idx)
                        if (edge_idx == 0):
                            if (len(aff_prob_shape) > 0 and len(other_shape) > 0):
                                error = error or not equal_shape(aff_prob_shape, other_shape)
                            elif (len(other_shape) > 0):
                                aff_prob_shape = other_shape
                        elif (edge_idx == 1):
                            if (len(aff_shape) > 0 and len(other_shape) > 0):
                                error = error or not equal_shape(aff_shape, other_shape)
                            elif (len(other_shape) > 0):
                                aff_shape = other_shape
                        elif (edge_idx == 2):
                            if (len(components) > 0 and len(other_shape) > 0):
                                error = error or not equal_shape(components, other_shape)
                            elif (len(other_shape) > 0):
                                components = other_shape
                
                    # Cross compare the shapes for validity according to the dimension rules for each shape
                    if (len(components) > 0):
                        error = error or not (len(components) > 2 and (components[1] == 1 or components[1] == 2))
                    if (len(components) > 0 and len(aff_shape) > 0 and len(aff_prob_shape) > 0):
                        error = error or not (len(components) == len(aff_shape) and len(components) == len(aff_prob_shape))
                    if (len(aff_shape) > 0 and len(aff_prob_shape) > 0):
                        error = error or not (equal_shape(aff_shape, aff_prob_shape))
                    if (len(aff_shape) > 0 and len(components) > 0):
                        error = error or not (equal_shape(aff_shape[2:], components[2:]))
                    if (len(aff_prob_shape) > 0 and len(components) > 0):
                        error = error or not (equal_shape(aff_prob_shape[2:], components[2:]))
            else:
                print('Unhandled sink: ' + sink.fn.type_name)
        return error
        
    def propagate_shape_forward(self, element, index, shape):
        existing = self.contains(element)
        if (type(element) == net_spec.Function):
            for suboutp in existing.out_edges:
                suboutp.set_shape(index, shape)
                if (len(suboutp.get_shape(index)) > 0):
                    for dst in suboutp.dsts:
                        dst.propagate_shape_forward(index)
        else:
            existing.set_shape(index, shape)
            if (len(existing.get_shape(index)) > 0):
                for dst in existing.dsts:
                    dst.propagate_shape_forward(index)
    
    def add_element(self, element):
        existing = self.contains(element)
        if (existing != None):
            return existing
        if (type(element) == net_spec.Function):
            node = Node(self, element)
            existing = node
        else:
            edge = Edge(self, element)
            existing = edge
        return existing
            
    def contains(self, element):
        for node in self.nodes:
            if (node.fn == element):
                return node
        for edge in self.edges:
            if (edge.top == element):
                return edge
        return None
        
    def get_srcs(self, function):
        srcs = []
        node = self.contains(function)
        if (node == None):
            node = self.add_element(function)
        srcs.append(node)
        return srcs
    
    def get_in_edges(self, inputs):
        edges = []
        for input in inputs:
            edge = self.contains(input)
            if (edge == None):
                edge = self.add_element(input)
            edges.append(edge)
        return edges
            
class Node:
    def __init__(self, graph, function):
        graph.nodes.append(self)
        self.fn = function
        self.graph = graph
        self.in_edges = []
        self.error = False
        if (isinstance(function, Iterable)):
            for subfunction in function:
                self.in_edges.extend(graph.get_in_edges(subfunction.inputs))
        else:
            self.in_edges.extend(graph.get_in_edges(function.inputs))
            
        self.out_edges = []
        
        for in_edge in self.in_edges:
            in_edge.dsts.append(self)
            
    def propagate_shape_forward(self, index):
        if (self.fn.type_name == 'Convolution'):
            pad = self.fn.params['pad'] if ('pad' in self.fn.params) else [0]
            stride = self.fn.params['stride'] if ('stride' in self.fn.params) else [1]
            dilation = self.fn.params['dilation'] if ('dilation' in self.fn.params) else [1]
            kernel_size = self.fn.params['kernel_size'] if ('kernel_size' in self.fn.params) else [1]
            num_output = self.fn.params['num_output'] if ('num_output' in self.fn.params) else [1]
                      
            for in_edge in self.in_edges:
                shape = copy.deepcopy(in_edge.get_shape(index))
                shape[1] = num_output
                for i in range(2,len(shape)):
                    j = i - 2
                    input_dim = shape[i]
                    kernel_extent = minidx(dilation, j) * (minidx(kernel_size, j) - 1) + 1
                    output_dim = (input_dim + 2 * minidx(pad, j) - kernel_extent) / minidx(stride, j) + 1
                    test_input_dim = ((output_dim - 1) * minidx(stride, j)) + kernel_extent - 2 * minidx(pad, j)
                    shape[i] = output_dim
                    
                    # Verify FW-BW shape conformity
                    if (not input_dim == test_input_dim):
                        self.error = True
                    
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
                break
        
        elif (self.fn.type_name == 'Deconvolution'):
            pad = self.fn.params['convolution_param']['pad'] if ('convolution_param' in self.fn.params and 'pad' in self.fn.params['convolution_param']) else [0]
            stride = self.fn.params['convolution_param']['stride'] if ('convolution_param' in self.fn.params and 'stride' in self.fn.params['convolution_param']) else [1]
            dilation = self.fn.params['convolution_param']['dilation'] if ('convolution_param' in self.fn.params and 'dilation' in self.fn.params['convolution_param']) else [1]
            kernel_size = self.fn.params['convolution_param']['kernel_size'] if ('convolution_param' in self.fn.params and 'kernel_size' in self.fn.params['convolution_param']) else [1]
            num_output = self.fn.params['convolution_param']['num_output'] if ('convolution_param' in self.fn.params and 'num_output' in self.fn.params['convolution_param']) else 1
                        
            for in_edge in self.in_edges:
                shape = copy.deepcopy(in_edge.get_shape(index))
                shape[1] = num_output
                for i in range(2,len(shape)):
                    j = i - 2
                    input_dim = shape[i]
                    kernel_extent = minidx(dilation, j) * (minidx(kernel_size, j) - 1) + 1
                    output_dim = ((input_dim - 1) * minidx(stride, j)) + kernel_extent - 2 * minidx(pad, j)
                    test_input_dim = (output_dim + 2 * minidx(pad, j) - kernel_extent) / minidx(stride, j) + 1
                    shape[i] = output_dim
                                       
                    # Verify FW-BW shape conformity
                    if (not input_dim == test_input_dim):
                        self.error = True
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
                break
        
        elif (self.fn.type_name == 'Pooling'):
            pad = self.fn.params['pad'] if ('pad' in self.fn.params) else [0]
            stride = self.fn.params['stride'] if ('stride' in self.fn.params) else [1]
            dilation = self.fn.params['dilation'] if ('dilation' in self.fn.params) else [1]
            kernel_size = self.fn.params['kernel_size'] if ('kernel_size' in self.fn.params) else [1]
            
            for in_edge in self.in_edges:
                shape = copy.deepcopy(in_edge.get_shape(index))
                for i in range(2,len(shape)):
                    j = i - 2
                    ext_kernel_shape = (minidx(kernel_size, j) - 1) * minidx(dilation, j) + 1
                    pooled_size = int(math.ceil(float(shape[i] + 2 * minidx(pad, j) - ext_kernel_shape) / minidx(stride, j))) + 1
                    test_size = (pooled_size - 1) * minidx(stride, j) + ext_kernel_shape - 2 * minidx(pad, j)                    
                    
                    # Verify FW-BW shape conformity
                    if (not shape[i] == test_size):
                        self.error = True
                    
                    if (minidx(pad, j) > 0):
                        if (pooled_size - 1) * minidx(stride, i) >= shape[i] + minidx(pad, j):
                            --pooled_size
                    shape[i] = pooled_size
                    
            if (len(shape) > 0):
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
            
        elif (self.fn.type_name == 'MergeCrop'):
            shape = []
            shape_A = self.in_edges[0].get_shape(index)
            shape_B = self.in_edges[1].get_shape(index)

            if (len(shape_A) > 0 and 'op' in self.fn.params and self.fn.params['op'] == 'add'):
                shape = copy.deepcopy(shape_A)
            elif (len(shape_A) > 0 and len(shape_B) > 0):
                shape = copy.deepcopy(shape_A)
                shape[1] = shape_A[1] + shape_B[1]
                
            if (len(shape_A) > 0 and len(shape_B) > 0):
                for i in range(2,len(shape_A)):
                    if (shape_A[i] > shape_B[i]):
                        self.error = True
            
            if (len(shape) > 0):
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
        
        elif (self.fn.type_name == 'Crop'):
            shape = []
                        
            shape_A = self.in_edges[0].get_shape(index)
            shape_B = self.in_edges[1].get_shape(index)
                       
            shape = copy.deepcopy(shape_B)
                        
            if (len(shape_A) > 0 and len(shape_B) > 0):
                for i in range(2,len(shape_A)):
                    if (shape_A[i] > shape_B[i]):
                        self.error = True
                        
            if len(shape) >= 2 and len(shape_A) >= 2:
                shape[0] = shape_A[0]
                shape[1] = shape_A[1]

            if (len(shape) > 0):
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
            
        elif (self.fn.type_name == 'InnerProduct'):
            num_output = self.fn.params['inner_product_param']['num_output'] if ('inner_product_param' in self.fn.params and 'num_output' in self.fn.params['inner_product_param']) else 1

            for in_edge in self.in_edges:
                shape = copy.deepcopy(in_edge.get_shape(index))
                shape[1] = num_output
                for i in range(2,len(shape)):
                    shape[i] = 1
                    
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, shape)
        # Shape stays the same
        else:
            for in_edge in self.in_edges:
                for out_edge in self.out_edges:
                    out_edge.set_shape(index, copy.deepcopy(in_edge.get_shape(index)))
                break
            
        # Propagate forward
        for out_edge in self.out_edges:
            if (len(out_edge.get_shape(index)) > 0):
                for dst in out_edge.dsts:
                    dst.propagate_shape_forward(index)
        
class Edge:
    def __init__(self, graph, top):
        graph.edges.append(self)
        self.top = top
        self.graph = None
        self.srcs = []
        self.error = False
        if (isinstance(top, Iterable)):
            for subtop in top:
                self.srcs.extend(graph.get_srcs(subtop.fn))
        else:
            self.srcs.extend(graph.get_srcs(top.fn))
        self.dsts = []
        self.shape = [[]]
        for src in self.srcs:
            src.out_edges.append(self)
    
    def get_shape(self, index):
        while (len(self.shape)- 1 < index):
            self.shape = self.shape + [[]]
        return copy.deepcopy(self.shape[index])
        
    def set_shape(self, index, shape):
        while (len(self.shape)- 1 < index):
            self.shape = self.shape + [[]]
        self.shape[index] = copy.deepcopy(shape)
        
    def check_shape_errors(self):
        error = False
        ref_shape = []
        for shape in self.shape:
            if (len(ref_shape) == 0):
                ref_shape = shape
            else:
                for i in range(0, min(len(ref_shape), len(shape))):
                    error = error or (not ref_shape[i] == shape[i])
                    error = error or (ref_shape[i] < 1 or shape[i] < 1)
        self.error = self.error or error

class Stack:
    def __init__(self):
        self.__storage = []
        
    def __len__(self):
        return len(self.__storage)

    def isEmpty(self):
        return len(self.__storage) == 0

    def push(self,p):
        self.__storage.append(p)

    def pop(self):
        return self.__storage.pop()
    
    
def minidx(data, index):
    return data[min(len(data) - 1, index)]

def equal_shape(shape_A, shape_B):
    equal = True
    if (not len(shape_A) == len(shape_B)):
        equal = False
    else:
        equal = True
        for i in range(0, len(shape_A)):
            equal = equal and shape_A[i] == shape_B[i]
    return equal

metalayers = MetaLayers()
