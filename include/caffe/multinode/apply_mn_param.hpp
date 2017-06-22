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

#ifndef _CAFFE_MULTINODE_APPLY_MN_PARAM_HPP_
#define _CAFFE_MULTINODE_APPLY_MN_PARAM_HPP_

#ifdef USE_MLSL

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

namespace caffe {
/**
 * @brief Apply the multinode parameters to the NetParameter
 *        inserting mn_activation layer if needed.
 */
template <typename Dtype>
void ApplyMultinodeParams(const NetParameter& param,
    NetParameter* param_with_mn);

/**
 * @brief Copy per-layer parameters from a Net object.
 */
template <typename Dtype>
void CopyMultinodeParamsFromNet(const Net<Dtype> *net, NetParameter *param);

/**
 * @brief Revert all the multinode changes from NetParameter
 */
template <typename Dtype>
void RevertMultinodeParams(NetParameter* param, bool write_diff = false);
}

#endif // USE_MLSL

#endif // _CAFFE_MULTINODE_APPLY_MN_PARAM_HPP_
