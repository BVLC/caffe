__author__ = 'pittnuts'
"""
prune layers in scnn convoluated with zeros
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
        if 'Convolution'==cur_layer.type and re.match("^conv.*q$",cur_layer.name):
            print 'generating:', cur_layer.name,cur_layer.type
            conv_layer_name = cur_layer.name[:-1]
            weights_t = src_net.params[ conv_layer_name ][0].data
            assert weights_t.shape[2]==1 and weights_t.shape[3]==1
            cur_m = cur_layer.convolution_param.group
            orig_grp_num = net_msg.layer._values[position_idx+1].convolution_param.group
            num_per_orig_grp = (cur_m/orig_grp_num)
            cur_sxs = weights_t.shape[1]*orig_grp_num/cur_m

            # add slice layer: slice output blobs of convxp
            conv_ptr = net_msg.layer.add()
            conv_ptr.name = conv_layer_name+"p_slice"
            conv_ptr.type = "Slice"
            conv_ptr.bottom.append(cur_layer.bottom._values[0])
            conv_ptr.slice_param.axis = 1
            for grp in range(0,cur_m):
                if grp!=0:
                    conv_ptr.slice_param.slice_point.append(grp)
                conv_ptr.top.append(conv_layer_name+"p_slice{}".format(grp))
            skept_q = []
            for grp in range(0,cur_m):
                orig_grp = grp/num_per_orig_grp
                nonzero_ratio = zeros((1,cur_sxs))
                outputs_grp_size = weights_t.shape[0]/orig_grp_num
                channel_idx = grp%num_per_orig_grp
                sub_bar = weights_t[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,channel_idx*cur_sxs:(channel_idx+1)*cur_sxs,:,:]
                sum_along_outputs = sum(abs(sub_bar),axis=0)
                cur_num_output = sum(sum_along_outputs>0)# nonzero slice
                if cur_num_output>0:
                    conv_ptr = net_msg.layer.add()
                    conv_ptr.CopyFrom(cur_layer)
                    #tmp_name = conv_ptr.name
                    conv_ptr.name = cur_layer.name + "_sub{}".format(grp)
                    conv_ptr.top._values[0] = conv_ptr.name#cur_layer.name
                    conv_ptr.bottom._values[0] = conv_layer_name+"p_slice{}".format(grp)
                    conv_ptr.convolution_param.num_output = cur_num_output
                    conv_ptr.convolution_param.group = 1
                else :
                    total_all_zero_counter += 1
                    skept_q.append(grp)

            # add Concatenation and split conv layer
            for orig_grp in range(0,orig_grp_num):
                #concatenation layer
                conv_ptr = net_msg.layer.add()
                conv_ptr.name = cur_layer.name+"_concat{}".format(orig_grp)
                for grp in range(0,num_per_orig_grp):
                    if (grp+orig_grp*num_per_orig_grp) not in skept_q:
                        conv_ptr.bottom.append(cur_layer.name + "_sub{}".format(grp+orig_grp*num_per_orig_grp))
                conv_ptr.top.append(conv_ptr.name)
                conv_ptr.type = "Concat"
                conv_ptr.concat_param.axis = 1
                #conv layer
                conv_ptr = net_msg.layer.add()
                conv_ptr.CopyFrom(net_msg.layer._values[position_idx+1])
                conv_ptr.name = conv_layer_name + "_split{}".format(orig_grp)
                conv_ptr.bottom._values[0] = cur_layer.name+"_concat{}".format(orig_grp)
                if orig_grp_num > 1:
                    conv_ptr.top._values[0] = conv_ptr.name
                conv_ptr.convolution_param.num_output = conv_ptr.convolution_param.num_output/orig_grp_num
                conv_ptr.convolution_param.group = 1

            #concatenation layer
            if orig_grp_num > 1:
                conv_ptr = net_msg.layer.add()
                conv_ptr.name = conv_layer_name+"_concat{}"
                for orig_grp in range(0,orig_grp_num):
                    conv_ptr.bottom.append(conv_layer_name+"_split{}".format(orig_grp))
                conv_ptr.top.append(conv_layer_name)
                conv_ptr.type = "Concat"
                conv_ptr.concat_param.axis = 1
            if orig_grp_num > 1:
                convxq_add_layers.append(1+cur_m+orig_grp_num*2+1-len(skept_q))
            else:
                convxq_add_layers.append(1+cur_m+orig_grp_num*2-len(skept_q))
            convxq_m.append(cur_m)
            convxq_positions.append(position_idx)

            #conv_ptr = net_msg.layer.add()
            #conv_ptr.name = cur_layer.name+"_concat"
            #for grp in range(0,cur_m):
            #    if grp>len(conv_ptr.bottom)-1:
            #        conv_ptr.bottom.append(cur_layer.name + "_sub{}".format(grp))
            #    else:
            #        conv_ptr.bottom[grp] = cur_layer.name + "_sub{}".format(grp)
            #conv_ptr.top.append(cur_layer.top[0])
            #conv_ptr.type = "Concat"
            #conv_ptr.concat_param.axis = 1#added_concat_param
            #
            #convxq_positions.append(position_idx)
            #convxq_m.append(cur_m+1)

            ## modifiy the channel number of layer t
            #conv_t = net_msg.layer.add()
            #conv_t = net_msg.layer._values.pop()
            #conv_t.CopyFrom(net_msg.layer._values[position_idx+1])
            #for grp in range(0,cur_m):
            #    if grp>len(conv_t.bottom)-1:
            #        conv_t.bottom.append(conv_t.name + "q_sub{}".format(grp))
            #    else:
            #        conv_t.bottom[grp] = conv_t.name + "q_sub{}".format(grp)
            ##if len(conv_t.bottom)==0:
            ##    conv_t.bottom.append(cur_layer.name)
            ##else:
            ##    conv_t.bottom[0] = cur_layer.name
            #net_msg.layer._values.pop(position_idx+1)
            #net_msg.layer._values.insert(position_idx+1,conv_t)

        #next layer
        position_idx += 1

    #reorder
    for pos,m_idx in zip(convxq_positions[::-1],convxq_add_layers[::-1]) :
        print pos,m_idx
        net_msg.layer._values.pop(pos) #pop convxq
        net_msg.layer._values.pop(pos) #pop convx
        l = len(net_msg.layer) #current length
        for grp in range(0,m_idx):
            net_msg.layer._values.insert(pos+grp,net_msg.layer._values[-m_idx+grp])
        for grp in range(0,m_idx):
            net_msg.layer._values.pop()
    # save prototxt
    file_split = os.path.splitext(srcproto)
    dstproto = file_split[0]+'_pruned'+file_split[1]
    file = open(dstproto, "w")
    if not file:
        raise IOError("ERROR (" + dstproto + ")!")
    file.write(str(net_msg))
    file.close()

    print net_msg
    #print convxq_positions
    #print convxq_m

####################################################################################################
    # open pruned net and generate caffemodel
    dst_net = caffe.Net(dstproto, caffe.TEST)
    print("src net:\n blobs {}\nparams {}\n".format(src_net.blobs.keys(), src_net.params.keys()))
    print("dst net:\n blobs {}\nparams {}\n".format(dst_net.blobs.keys(), dst_net.params.keys()))

    counter = 0
    for param_key in src_net.params.keys():
        if re.match("^conv[0-9]+q$",param_key):
            #assert len(src_net.params[param_key])==1
            #weights_convxq = src_net.params[param_key][0].data
            #weights_convx = src_net.params[param_key][0].data
            pass
        elif re.match("^conv[0-9]+$",param_key):
            convq_param_key = param_key+"q"
            assert len(src_net.params[convq_param_key])==1
            weights_convxq = src_net.params[convq_param_key][0].data
            weights_convx = src_net.params[param_key][0].data
            bias_convx = src_net.params[param_key][1].data
            weights_bias = src_net.params[param_key][1].data

            orig_grp_num = weights_convxq.shape[0]/weights_convx.shape[1]
            cur_m = convxq_m[counter]
            num_per_orig_grp = (cur_m/orig_grp_num)
            cur_sxs = weights_convx.shape[1]*orig_grp_num/cur_m
            counter += 1
            for orig_grp in range(0,orig_grp_num):
                outputs_grp_size = weights_convx.shape[0]/orig_grp_num
                for _grp in range(0,num_per_orig_grp):
                    grp = _grp + orig_grp*num_per_orig_grp
                    nonzero_ratio = zeros((1,cur_sxs))
                    channel_idx = grp%num_per_orig_grp
                    sub_bar = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,channel_idx*cur_sxs:(channel_idx+1)*cur_sxs,:,:]
                    sum_sub_bar = sum(abs(sub_bar),axis=0)>0
                    num_outputs = sum(sum_sub_bar)
                    sublayer_name = param_key + "q_sub{}".format(grp)
                    if num_outputs==0:
                        print "{} is skept".format(sublayer_name)
                    else:
                        nonzero_i = nonzero(sum_sub_bar.flatten())[0] + grp*cur_sxs
                        assert num_outputs == dst_net.params[sublayer_name][0].data.shape[0]
                        src_weights = weights_convxq[nonzero_i,:,:,:]
                        dst_weights = dst_net.params[sublayer_name][0].data
                        assert dst_weights.shape == src_weights.shape
                        dst_weights[:] = src_weights#weights_convxq[nonzero_i,:,:,:]
                #convx weights
                nonzero_ratio = zeros((1,cur_sxs*num_per_orig_grp))
                sub_bar = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,:,:,:]
                sum_sub_bar = sum(abs(sub_bar),axis=0)>0
                num_inputs = sum(sum_sub_bar)
                nonzero_i = nonzero(sum_sub_bar.flatten())[0]
                dst_layer_name = param_key+"_split{}".format(orig_grp)
                assert num_inputs == dst_net.params[dst_layer_name][0].data.shape[1]
                src_weights = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,nonzero_i,:,:]
                dst_weights = dst_net.params[dst_layer_name][0].data
                assert dst_weights.shape == src_weights.shape
                dst_weights[:] = src_weights
                #convx bias
                dst_net.params[dst_layer_name][1].data[:] = bias_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size]


            #for grp in range(0,cur_m):
            #    orig_grp = grp/num_per_orig_grp
            #    nonzero_ratio = zeros((1,cur_sxs))
            #    outputs_grp_size = weights_convx.shape[0]/orig_grp_num
            #    channel_idx = grp%num_per_orig_grp
            #    sub_bar = weights_convx[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,channel_idx*cur_sxs:(channel_idx+1)*cur_sxs,:,:]
            #    sum_sub_bar = sum(abs(sub_bar),axis=0)>0
            #    num_outputs = sum(sum_sub_bar)
            #    sublayer_name = param_key + "q_sub{}".format(grp)
            #    if num_outputs==0:
            #        print "{} is skept".format(sublayer_name)
            #    else:
            #        nonzero_i = nonzero(sum_sub_bar.flatten())[0]
            #        assert num_outputs == dst_net.params[sublayer_name][0].data.shape[0]
            #        src_weights = weights_convxq[nonzero_i,:,:,:]
            #        dst_weights = dst_net.params[sublayer_name][0].data
            #        assert dst_weights.shape == src_weights.shape
            #        dst_weights[:] = weights_convxq[nonzero_i,:,:,:]

            #for orig_grp in range(0,orig_grp_num):
            #    #conv layer
            #    conv_ptr = net_msg.layer.add()
            #    conv_ptr.CopyFrom(net_msg.layer._values[position_idx+1])
            #    conv_ptr.name = conv_layer_name + "_split{}".format(orig_grp)
            #    conv_ptr.bottom._values[0] = cur_layer.name+"_concat{}".format(orig_grp)
            #    if orig_grp_num > 1:
            #        conv_ptr.top._values[0] = conv_ptr.name
            #    conv_ptr.convolution_param.num_output = conv_ptr.convolution_param.num_output/orig_grp_num
            #    conv_ptr.convolution_param.group = 1
        else:
            dst_net.params[param_key][:] = src_net.params[param_key][:]
    #net_msg = src_net_parser.readProtoNetFile()
    #layer_idx = 0
    #loop_layers = net_msg.layer[:]
    #position_idx = 0
    #for cur_layer in loop_layers:
    #    if 'Convolution'==cur_layer.type and re.match("^conv.*q$",cur_layer.name):
    #        print 'generating:', cur_layer.name,cur_layer.type
    #        conv_layer_name = cur_layer.name[:-1]
    #        weights_t = src_net.params[ conv_layer_name ][0].data
    #        assert weights_t.shape[2]==1 and weights_t.shape[3]==1
    #        cur_m = cur_layer.convolution_param.group
    #        orig_grp_num = net_msg.layer._values[position_idx+1].convolution_param.group
    #        num_per_orig_grp = (cur_m/orig_grp_num)
    #        cur_sxs = weights_t.shape[1]*orig_grp_num/cur_m
    #        for grp in range(0,cur_m):
    #            orig_grp = grp/num_per_orig_grp
    #            conv_ptr = net_msg.layer.add()
    #            conv_ptr.CopyFrom(cur_layer)
    #            #tmp_name = conv_ptr.name
    #            conv_ptr.name = cur_layer.name + "_sub{}".format(grp)
    #            conv_ptr.top._values[0] = conv_ptr.name#cur_layer.name
    #            nonzero_ratio = zeros((1,cur_sxs))
    #            outputs_grp_size = weights_t.shape[0]/orig_grp_num
    #            channel_idx = grp%num_per_orig_grp
    #            sub_bar = weights_t[orig_grp*outputs_grp_size:(orig_grp+1)*outputs_grp_size,channel_idx*cur_sxs:(channel_idx+1)*cur_sxs,:,:]
    #            sum_along_outputs = sum(abs(sub_bar),axis=0)
    #            conv_ptr.convolution_param.num_output = sum(sum_along_outputs>0)# nonzero slice
    #            if conv_ptr.convolution_param.num_output==0:
    #                pass
    #            conv_ptr.convolution_param.group = 1



    print "total_all_zero_counter = {}".format(total_all_zero_counter)
    file_split = os.path.splitext(srcmodel)
    dstmodel = file_split[0]+'_pruned'+file_split[1]
    dst_net.save(dstmodel)
    print 'Done.'