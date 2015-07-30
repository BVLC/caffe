#!/usr/bin/env python
"""
Convert a fully-connected net to a fully-convolutional one
"""
import numpy as np
import matplotlib.pyplot as plt
import Image

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import caffe

def parse_args():
	parser = ArgumentParser(description=__doc__,
                          formatter_class=ArgumentDefaultsHelpFormatter)

	parser.add_argument('input_net_proto_file',
                      help='Input network prototxt file')
	parser.add_argument('input_net_proto_conv_file',
                      help='Input fully conv network prototxt file')
	parser.add_argument('input_caffemodel_file',
                      help='Input network caffemodel file')
	parser.add_argument('output_caffemodel_file',
                      help='Output .caffemodel file')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	caffe.set_mode_cpu()
	net = caffe.Net(args.input_net_proto_file, caffe.TEST)
	print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

	# Load the original network and extract the fully connected layers' parameters.
	net = caffe.Net(args.input_net_proto_file,
									args.input_caffemodel_file,
									caffe.TEST)

	params = ['fc_200_left_L2', 'fc_200_right_L2', 'fc_200_left_L3',
						'fc_200_right_L3', 'fc_300_L4', 'fc_300_L5',
						'fc_300_L6', 'fc_300_L7', 'fc_2_L8']

	fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

	for fc in params:
			print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

	# Load the fully convolutional network to transplant the parameters.
	net_full_conv = caffe.Net(args.input_net_proto_conv_file,
														args.input_caffemodel_file,
														caffe.TEST)

	params_full_conv = ['fc_200_left_L2-conv', 'fc_200_right_L2-conv',
											'fc_200_left_L3-conv', 'fc_200_right_L3-conv',
											'fc_300_L4-conv', 'fc_300_L5-conv', 'fc_300_L6-conv',
											'fc_300_L7-conv', 'fc_2_L8-conv']

	conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

	for conv in params_full_conv:
			print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

	# copy the weights from the fully connected version to the fully convolution one
	for pr, pr_conv in zip(params, params_full_conv):
			conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
			conv_params[pr_conv][1][...] = fc_params[pr][1]

	net_full_conv.save(args.output_caffemodel_file)

if __name__ == '__main__':
	main()
