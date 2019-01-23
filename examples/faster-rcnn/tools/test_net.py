#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.test import sample_net
from fast_rcnn import net_sample_utils
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe

import time, os, sys
utilpath = os.path.split(os.path.realpath(__file__))[0] + "/../../../scripts/"
sys.path.insert(0, utilpath)
import calibrator
import sampling
import argparse
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--quantized_net', action='store', dest='quantized_prototxt',
			default=None, type=str)
    parser.add_argument('--sample_iters', action='store', dest='sample_iters',
                        default=100, type=int)
    parser.add_argument('--quant_mode', action='store', dest='quant_mode',
			default='single', type=str)

    parser.add_argument('-u', '--unsigned_range', dest='unsigned_range', action="store_true", default=False,
                        help='to quantize using unsigned range for activation')

    parser.add_argument('-t', '--concat_use_fp32', dest='concat_use_fp32', action="store_true", default=False,
                        help='to use fp32 for concat')

    parser.add_argument('-f', '--unify_concat_scales', dest='unify_concat_scales', action="store_true", default=False,
                        help='to unify concat scales')

    parser.add_argument('-a', '--calibration_algos', dest='calibration_algos', action='store', default="DIRECT",
                        help='to choose the calibration alogorithm')

    parser.add_argument('-wi', '--conv_algo', dest='conv_algo', action="store_true", default=False,
                        help='to choose the convolution algorithm')
    parser.add_argument('-1st', '--enable_1st_conv_layer', dest='enable_1st_conv_layer', action="store_true", default=False,
                        help='enable 1st conv layer')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    #caffe.set_mode_gpu()
    #caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    if args.quantized_prototxt == None:
	test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)
    else:
        (blobs, params, top_blobs_map, bottom_blobs_map, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, winograd_convolutions) = sample_net(args.prototxt, net, imdb, args.sample_iters, args.quant_mode, args.enable_1st_conv_layer)

        (inputs_max, outputs_max, inputs_min) = sampling.calibrate_activations(blobs, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, args.calibration_algos, "SINGLE", args.conv_algo)
        params_max = sampling.calibrate_parameters(params, winograd_convolutions, "DIRECT", args.quant_mode.upper(), args.conv_algo)
        calibrator.generate_sample_impl(args.prototxt, args.quantized_prototxt, inputs_max, outputs_max, inputs_min, params_max, args.enable_1st_conv_layer)
        compiled_net_str = caffe.compile_net(args.prototxt, caffe.TEST, "MKLDNN")
        raw_net_basename = os.path.basename(args.prototxt)
        compile_net_path = "./compiled_" + raw_net_basename
        with open(compile_net_path, "w") as f:
            f.write(compiled_net_str)
        calibrator.transform_convolutions(args.quantized_prototxt, compile_net_path, top_blobs_map, bottom_blobs_map, args.unsigned_range, args.concat_use_fp32, args.unify_concat_scales, args.conv_algo, args.enable_1st_conv_layer)
