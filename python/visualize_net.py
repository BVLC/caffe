
import os
import sys
import caffe
import caffe.io
import argparse
import shutil
import numpy as np
from PIL import Image

#np.set_printoptions(precision=8, linewidth=200, edgeitems=50)

def init_network(args):
	net = caffe.Net(args.net_file, args.weights_file, caffe.TEST)
	for name in net.blobs:
		net.blobs[name].reshape(1, *(net.blobs[name].data.shape[1:]))
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
	transformer.set_raw_scale('data', 255)
	#transformer.set_channel_swap('data', (2,1,0))
	net.transformer = transformer
	caffe.set_mode_cpu()
	return net


def print_arch(net):
	def prod(l):
		p = 1
		for x in l:
			p *= x
		return p
	print "Blobs:"
	for name, blob in net.blobs.items():
		print "\t%s: %s" % (name, blob.data.shape)
	print

	num_params = 0
	print "Parameters:"
	for name, lblob in net.params.items():
		num_param = sum(map(lambda blob: prod(blob.data.shape), lblob))
		print "\t%s: %s\t%d" % (name, "\t".join(map(lambda blob: str(blob.data.shape), lblob)), num_param)
		num_params += num_param
	print
	print "Num Parameters:", num_params
	print

	print "Inputs:"
	for name in net.inputs:
		print "\t%s" % name
	print

	print "Outputs:"
	for name in net.outputs:
		print "\t%s" % name
	print

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def format_data(data, padsize=1, padval=0):
	data -= data.min()
	data /= data.max()
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	return data

_original = False
def format_general_filters(data, padsize=1, padval=0):
	num_filters = data.shape[0]
	num_channels = data.shape[1]
	if _original:
		return format_data(data[:num_channels].reshape( (num_channels ** 2,) + data.shape[2:]))
	
	data -= data.min()
	data /= data.max()
	padding = ( (0, 0), (0, 0), (0, padsize), (0, padsize) )
	data = np.pad(data, padding, mode='constant', constant_values=padval)
	
	data = data.reshape((num_filters, num_channels) + data.shape[2:]).transpose((0, 2, 1, 3))
	data = data.reshape((num_filters * data.shape[1], num_channels * data.shape[3]))
	return data
	

def save_blob(blob, out_file, args):
	if len(blob.shape) == 1:
		blob = np.reshape(blob, (1, blob.shape[0]))
	if len(blob.shape) == 4:
		num_filters = blob.shape[0]
		num_channels = blob.shape[1]
		width = blob.shape[2]
		height = blob.shape[3]

		if num_channels == 3:
			# RGB
			np_im = format_data(blob.transpose(0, 2, 3, 1))
		else:
			# this works if the number of channels is <= number of filters
			np_im = format_general_filters(blob)
	elif len(blob.shape) == 3:
		num_channels = blob.shape[0]
		if num_channels == 3:
			blob = np.reshape(blob, (1,) + blob.shape)
			np_im = format_data(blob.transpose(0, 2, 3, 1))
		else:
			np_im = format_data(blob, padval=1)
	else:
		np_im = blob - blob.min()
		np_im /= np_im.max()
	im = Image.fromarray(((255 * np_im).astype("uint8")))
	im = im.resize( (2 * im.size[0], 2 * im.size[1]), resample=Image.NEAREST)
	im.save(out_file)
	

def save_filters(net, args):
	out_dir = os.path.join(args.output_dir, "filters")
	os.mkdir(out_dir)
	for name, lblob in net.params.items():
		print name
		out_file = os.path.join(out_dir, name + ".png")
		weights = lblob[0].data
		save_blob(weights, out_file, args)

def save_activations(net, args):
	for im_f in args.test_images:
		print im_f
		outdir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(im_f))[0])
		os.mkdir(outdir)
		im = caffe.io.load_image(im_f)
		net.blobs['data'].data[...] = net.transformer.preprocess('data', im)
		net.forward()
		for name, data in net.blobs.items():
			try:
				print "\t%s: %s" % (name, data.data.shape)
				out_file = os.path.join(outdir, name + ".png")
				if len(data.data.shape) > 1:
					save_blob(data.data[0], out_file, args)
			except Exception as e:
				print "skipping %s" % name

def main(args):
	print "Initializing Network"
	net = init_network(args)
	print "Architecture"
	print_arch(net)
	if os.path.exists(args.output_dir):
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir)

	print "\nSaving Activations"
	save_activations(net, args)

	print "\nSaving Filters"
	save_filters(net, args)

def get_args():
	parser = argparse.ArgumentParser(description="Output network activations and filters")
	parser.add_argument('net_file', 
		help="prototxt file containing the network definition")
	parser.add_argument('weights_file', 
		help="prototxt file containing the network weights for the given definition")
	parser.add_argument('output_dir', 
		help="path to the output directory where images are written")
	parser.add_argument('test_images', nargs=argparse.REMAINDER,
		help="images to run through the network")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()
	main(args)

