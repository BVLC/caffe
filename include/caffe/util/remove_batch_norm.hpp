
/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef COMPILE_NET_UTIL_HPP_
#define COMPILE_NET_UTIL_HPP_
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 *  @brief If CompileNet's compilation rule one does work, some scale layer's weights and bias blobs
 *  may be merged into batch norm layer. RecoverScaleFromBN will recover the merged scale layer's info.
 *  Currently, we only care about the weights and bias info.
 */
template <typename Dtype>
void RecoverScaleFromBN(const LayerParameter& bn_layer_param, LayerParameter& scale_layer_param, Dtype default_scale_weights, Dtype default_scale_bias);
/**
 *  @brief rename layer1's top to layer2's
 */
void MergeLayer(LayerParameter &layer1, const LayerParameter &layer2);

/**
 *  @brief After removing the batch norm and scale layer after a convolution layer, to make the inference
 *  result correct, we must adjust convolution layer's weights and bias blobs
 */

template <typename Dtype>
void AdjustConvLayer(LayerParameter &conv_layer,
                     const LayerParameter &batch_norm_layer,
                     const LayerParameter &scale_layer, bool is_net_init);

/**
 *  @brief The batch norm and scale layer may be merged due to compilation rule one's effect, RecoverBNScaleMergedNet
 *  is used to recover the scale layer
 */
template <typename Dtype>
void RecoverBNScaleMergedNet(NetParameter * net_param, NetParameter* recovered_net_param);

}
#endif
