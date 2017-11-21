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

#if defined (USE_MLSL) && defined (ENABLE_WEIGHT_GRAD_COMPRESSION)

#include <dlfcn.h>
#include <string.h>

#include "dl_compression.h"
#include "caffe/util/compression_util.hpp"
#include "caffe/common.hpp"
#include "caffe/multinode/mlsl.hpp"


namespace caffe {

void get_weight_grad_compress_info()
{
  MLSL::QuantParams param;
  Dl_info info;

  if (!dl_comp_check_running_environ()) {
    LOG(INFO) << "Disable weight grad compression because "
              << "running environment does not meet requiremnt "
              << "e.g. avx512 instruction is not supported.";
    return;
  }
 
  int ret = dladdr((void *)dl_comp_compress_buffer, &info);
  if (ret == 0) {
    LOG(FATAL) << "Get Compress Lib Info Failed!\n";
  }
  param.quant_buffer_func_name = strdup(info.dli_sname);
  param.lib_path = strdup(info.dli_fname);

  ret = dladdr((void *)dl_comp_decompress_buffer, &info);
  if (ret == 0) {
    LOG(FATAL) << "Get Compress Lib Info Failed!\n";
  }
  param.dequant_buffer_func_name = strdup(info.dli_sname);

  ret = dladdr((void *)dl_comp_compressed_buffer_reduce_sum, &info);
  if (ret == 0) {
    LOG(FATAL) << "Get Compress Lib Info Failed!\n";
  }
  param.reduce_sum_func_name = strdup(info.dli_sname);

  param.block_size = dl_comp_get_sizeof_block(DL_COMP_FLOAT32, 4, DL_COMP_DFP);
  param.elem_in_block = dl_comp_get_elem_num_in_block();
 
  mn::train::set_quantization_param(&param);

  free(param.quant_buffer_func_name);
  free(param.lib_path);
  free(param.dequant_buffer_func_name);
  free(param.reduce_sum_func_name);
}

}

#endif // USE_MLSL
