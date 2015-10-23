__author__ = 'pittnuts'

'''
generate caffemodel for equivalent scnn
'''

import argparse
import caffeparser
import caffe
from caffe.proto import caffe_pb2
from numpy import *
import re
import pittnuts
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

# caffemodel
# --srcproto models/bvlc_reference_caffenet/deploy.prototxt
# --srcmodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
# --dstproto models/eilab_reference_sparsenet/deploy_scnn.prototxt
# --dstmodel models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel

# cifar quick model
# --srcproto examples/cifar10/cifar10_quick.prototxt
# --srcmodel examples/cifar10/cifar10_quick_iter_5000.caffemodel
# --dstproto examples/cifar10/cifar10_quick_scnn.prototxt
# --dstmodel examples/cifar10/eilab_cifar10_quick_ini_sparsenet.caffemodel


    parser.add_argument('--srcproto', type=str, required=True)
    parser.add_argument('--srcmodel', type=str, required=True)
    parser.add_argument('--dstproto', type=str, required=True)
    parser.add_argument('--dstmodel', type=str, required=True)
    args = parser.parse_args()
    srcproto = args.srcproto
    srcmodel = args.srcmodel
    dstproto = args.dstproto
    dstmodel = args.dstmodel
    print args

    caffe.set_device(0)
    caffe.set_mode_gpu()

    src_net = caffe.Net(srcproto,srcmodel, caffe.TEST)
    #src_net_parser = caffeparser.CaffeProtoParser(dstproto)
    #src_net_layer_info = src_net_parser.readProtoNetFile()
    dst_net = caffe.Net(dstproto, caffe.TEST)
    print("src net:\n blobs {}\nparams {}\n".format(src_net.blobs.keys(), src_net.params.keys()))
    print("dst net:\n blobs {}\nparams {}\n".format(dst_net.blobs.keys(), dst_net.params.keys()))

    for layer_name in src_net.params.keys():
        if re.match("^conv.*",layer_name):
            print '\nfilling parameters for {}'.format(layer_name)
            weights = src_net.params[layer_name][0].data
            bias = src_net.params[layer_name][1].data
            print "src weight {}, bias {}".format(weights.shape, bias.shape)
            print "dst weight, bias"
            weights_p = dst_net.params[layer_name+'p'][0].data
            weights_q = dst_net.params[layer_name+'q'][0].data
            weights_s = dst_net.params[layer_name][0].data
#            bias_p = dst_net.params[layer_name+'p'][1].data
#            bias_q = dst_net.params[layer_name+'q'][1].data
            bias_s = dst_net.params[layer_name][1].data
            print "P {} {}".format(weights_p.shape,0)
            print "Q {} {}".format(weights_q.shape,0)
            print "S {} {}".format(weights_s.shape,bias_s.shape)
            weights = transpose(weights,(2,3,1,0))
            group = weights_p.shape[0]/weights_p.shape[1]
            group_size = weights.shape[3]/group
            group_p_size = weights_p.shape[0]/group
            group_q_size = weights_q.shape[0]/group
            group_s_size = weights_s.shape[0]/group
            print "{} group(s) ".format(group)
            for g in range(0,group):
                group_weights = weights[:,:,:,g*group_size:(g+1)*group_size]
                P,S,Q,qi = pittnuts.kernel_factorization(group_weights)
                weights_p[g*group_p_size:(g+1)*group_p_size,:,0,0] = P
                weights_q[g*group_q_size:(g+1)*group_q_size,:,:,:] = Q.transpose((0,3,1,2)).reshape((group_q_size,weights_q.shape[1],weights_q.shape[2],weights_q.shape[3]))
                weights_s[g*group_s_size:(g+1)*group_s_size,:,:,:] = S.transpose(2,0,1).reshape((group_s_size,weights_s.shape[1],weights_s.shape[2],weights_s.shape[3]))
            bias_s[:] = bias
        elif re.match("^fc.*",layer_name) or re.match("^ip.*",layer_name):
            print "filling {}".format(layer_name)
            dst_net.params[layer_name][0].data[:] = src_net.params[layer_name][0].data[:]
            dst_net.params[layer_name][1].data[:] = src_net.params[layer_name][1].data[:]





    #dst_net_parser = caffeparser.CaffeProtoParser(dstproto)
    #dst_net_param = dst_net_parser.readProtoNetFile()

    #print(dst_net_msg)
# --srcproto models/bvlc_reference_caffenet/train_val.prototxt
# --srcmodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
# --dstproto models/eilab_reference_sparsenet/train_val_scnn.prototxt
# --dstmodel models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel
    #read source caffemodel
    #f = open(srcmodel,'rb')
    #src_net_param = caffe_pb2.NetParameter()
    #src_net_param.ParseFromString(f.read())
    #f.close()
    ##data = array(caffe.io.blobproto_to_array(blob))

    ##edit source caffemodel
    #print blob.ListFields()
    #print blob.ByteSize()

    #save to dst caffemodel
    #f = open(dstmodel,'wb')
    #f.write(dst_net_param.SerializeToString())
    #f.close()

    #f = open(dstmodel,'rb')
    #dst_net_param_2 = caffe_pb2.NetParameter()
    #dst_net_param_2.ParseFromString(f.read())
    #f.close()
    dst_net.save(dstmodel)
    print 'Done.'