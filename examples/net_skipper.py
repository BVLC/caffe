__author__ = 'pittnuts'
"""
add skipped_channels and skipped_outputs into convxq and convx layers in scnn to avoid convoluating with zeros
"""

import argparse
import caffeparser
import caffe
from caffe.proto import caffe_pb2
from numpy import *
import re
import pittnuts
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#--srcproto examples/cifar10/cifar10_full_scnn.prototxt --srcmodel examples/cifar10/cifar10_full_scnn_iter_120000_zerout.caffemodel

    parser.add_argument('--srcproto', type=str, required=True)
    parser.add_argument('--srcmodel', type=str, required=True)
    #parser.add_argument('--dstproto', type=str, required=False)
    #parser.add_argument('--dstmodel', type=str, required=False)
    args = parser.parse_args()
    srcproto = args.srcproto
    srcmodel = args.srcmodel
    #dstproto = args.dstproto
    #dstmodel = args.dstmodel

    caffe.set_device(0)
    caffe.set_mode_gpu()

    src_net = caffe.Net(srcproto,srcmodel, caffe.TEST)
    print("src net:\n blobs {}\nparams {}\n".format(src_net.blobs.keys(), src_net.params.keys()))
    src_net_parser = caffeparser.CaffeProtoParser(srcproto)
    net_msg = src_net_parser.readProtoNetFile()

    layer_idx = 0
    loop_layers = net_msg.layer[:] #adding : implicitly makes a copy to avoid being modified in the loop
    convxq_positions = []
    convxq_m = []
    convxq_add_layers = []
    position_idx = 0

    total_all_zero_counter = 0

    # generate and save dst prototxt

    for cur_layer in loop_layers:
        if 'Convolution'==cur_layer.type and re.match("^conv[0-9]+$",cur_layer.name):
            convq_layer = net_msg.layer._values[position_idx-1]
            convq_param_key = cur_layer.name+"q"
            param_key = cur_layer.name
            convx_ptr = net_msg.layer._values.pop(position_idx)
            convx_ptr.CopyFrom(cur_layer)
            convxq_ptr = net_msg.layer._values.pop(position_idx-1)
            convxq_ptr.CopyFrom(convq_layer)

            assert len(src_net.params[convq_param_key])==1
            weights_convxq = src_net.params[convq_param_key][0].data
            weights_convx = src_net.params[param_key][0].data
            assert weights_convx.shape[3]==1 and weights_convx.shape[2]==1

            orig_grp_num = weights_convxq.shape[0]/weights_convx.shape[1]
            cur_m = convq_layer.convolution_param.group
            orig_grp_num = cur_layer.convolution_param.group
            num_per_orig_grp = (cur_m/orig_grp_num)
            cur_sxs = weights_convx.shape[1]*orig_grp_num/cur_m
            forward_counter = 0
            for orig_grp in range(0,orig_grp_num):
                outputs_grp_size = weights_convx.shape[0]/orig_grp_num
                #for _grp in range(0,num_per_orig_grp):
                #    grp = _grp + orig_grp*num_per_orig_grp
                #    nonzero_ratio = zeros((1,cur_sxs))
                #    channel_idx = grp%num_per_orig_grp
                #    sub_bar = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,channel_idx*cur_sxs:(channel_idx+1)*cur_sxs,:,:]
                #    sum_sub_bar = sum(abs(sub_bar),axis=0)==0
                #    skipped_num = sum(sum_sub_bar)
                #    sublayer_name = param_key + "q_sub{}".format(grp)
                #    if skipped_num==cur_sxs:
                #        print "{} is completely skept".format(sublayer_name)
                #    #if skipped_num>0:
                #    #    skipped_i = nonzero(sum_sub_bar.flatten())[0] + grp*cur_sxs
                #    #    assert skipped_num == dst_net.params[sublayer_name][0].data.shape[0]
                #    #    src_weights = weights_convxq[skipped_i,:,:,:]
                #    #    dst_weights = dst_net.params[sublayer_name][0].data
                #    #    assert dst_weights.shape == src_weights.shape
                #    #    dst_weights[:] = src_weights#weights_convxq[skipped_i,:,:,:]

                nonzero_ratio = zeros((1,cur_sxs*num_per_orig_grp))
                sub_bar = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,:,:,:]
                sum_sub_bar = sum(abs(sub_bar),axis=0)>0
                #num_inputs = sum(sum_sub_bar)
                forward_i = nonzero(sum_sub_bar.flatten())[0] + cur_sxs*num_per_orig_grp*orig_grp
                #dst_layer_name = param_key+"_split{}".format(orig_grp)
                #assert num_inputs == dst_net.params[dst_layer_name][0].data.shape[1]
                #src_weights = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,forward_i,:,:]
                #dst_weights = dst_net.params[dst_layer_name][0].data
                #assert dst_weights.shape == src_weights.shape
                #dst_weights[:] = src_weights
                for s in forward_i:
                    convxq_ptr.convolution_param.forward_outputs.append(s)
                    convx_ptr.convolution_param.forward_channels.append(s)
                forward_counter += len(forward_i)

            net_msg.layer._values.insert(position_idx-1,convxq_ptr)
            net_msg.layer._values.insert(position_idx,convx_ptr)
            print "{}:{}".format(cur_layer.name,1-forward_counter/(float)(cur_sxs*cur_m))

        position_idx += 1



    # save prototxt
    file_split = os.path.splitext(srcproto)
    dstproto = file_split[0]+'_skipped'+file_split[1]
    file = open(dstproto, "w")
    if not file:
        raise IOError("ERROR (" + dstproto + ")!")
    file.write(str(net_msg))
    file.close()

    #print net_msg

