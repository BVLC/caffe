#include <cstddef>
#include <iostream>
#include <algorithm>

#include "caffe/3rdparty/ctc/ctc.h"
#include "caffe/3rdparty/ctc/detail/cpu_ctc.cuh"
#ifdef __CUDACC__
#include "caffe/3rdparty/ctc/detail/gpu_ctc.cuh"
#endif


extern "C" {

ctcStatus_t compute_ctc_loss_gpu(const float* const activations,
                                 float* gradients,
                                 const int* const flat_labels,
                                 const int* const label_lengths,
                                 const int* const input_lengths,
                                 int alphabet_size,
                                 int minibatch,
                                 float *costs,
                                 void *workspace,
                                 ctcOptions options) {
#ifdef __CUDACC__
        GpuCTC<float> ctc(alphabet_size, minibatch, workspace, options.stream,
                          options.blank_label);

        if (gradients != NULL)
            return ctc.cost_and_grad(activations, gradients, costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
        else
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return CTC_STATUS_EXECUTION_FAILED;
#endif
}
}
