# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import caffe
import google.protobuf.text_format as txtf
from caffe.proto import caffe_pb2

def update_conv_quantized_dict(conv_quantized_dict, tmp_conv_quantized_dict):
    for conv in conv_quantized_dict:
        if tmp_conv_quantized_dict[conv][0] > conv_quantized_dict[conv][0]:
            conv_quantized_dict[conv][0] = tmp_conv_quantized_dict[conv][0]

        if tmp_conv_quantized_dict[conv][1] > conv_quantized_dict[conv][1]:
            conv_quantized_dict[conv][1] = tmp_conv_quantized_dict[conv][1]


def create_quantized_net(raw_net, quantized_net, conv_quantized_dict):
    net_param = caffe_pb2.NetParameter()
    with open(raw_net) as f:
        txtf.Merge(f.read(), net_param)
    #skip first conv layer when quantizing net
    first_conv = True
    for layer_param in net_param.layer:
        if layer_param.type == "Convolution":
            if first_conv:
                first_conv = False
                continue
            layer_param.quantization_param.bw_layer_in = 8
            layer_param.quantization_param.bw_layer_out = 8
            layer_param.quantization_param.bw_params = 8
            layer_param.quantization_param.scale_in.append(conv_quantized_dict[layer_param.name][0])
            layer_param.quantization_param.scale_out.append(conv_quantized_dict[layer_param.name][1])
            for param_scale in conv_quantized_dict[layer_param.name][2]:
                layer_param.quantization_param.scale_params.append(param_scale)
    with open(quantized_net, 'w') as f:
        f.write(str(net_param))
