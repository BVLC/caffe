#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

#ifdef USE_HALF
  if (std::is_same<MItype, half_fp>::value) {
    ss << "#define DTYPE_MAX HALF_MAX" << std::endl;
    ss << "#define DTYPE_MIN HALF_MIN" << std::endl;
  } else if (std::is_same<MItype, float>::value
        || std::is_same<MItype, double>::value) {
#endif
    ss << "#define DTYPE_MAX FLT_MAX" << std::endl;
    ss << "#define DTYPE_MIN FLT_MIN" << std::endl;
#ifdef USE_HALF
  } else {
    ss << "#define DTYPE_MAX " << 0 << std::endl;
    ss << "#define DTYPE_MIN " << 0 << std::endl;
  }
#endif

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolForwardSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "int_tp pw = index % pooled_width;" << std::endl;
    ss << "int_tp ph = (index / pooled_width) % pooled_height;" << std::endl;
    ss << "int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "int_tp hend = min((int_tpc) (hstart + ext_kernel_h),"
       << " (int_tpc) height);" << std::endl;
    ss << "int_tp wend = min((int_tpc) (wstart + ext_kernel_w),"
       << " (int_tpc) width);" << std::endl;
    ss << "while (hstart < 0) {" << std::endl;
    ss << "hstart += dilation_h;" << std::endl;
    ss << "}" << std::endl;
    ss << "while (wstart < 0) {" << std::endl;
    ss << "wstart += dilation_w;" << std::endl;
    ss << "}" << std::endl;
    ss << "Dtype maxval = -DTYPE_MAX;" << std::endl;
    ss << "int_tp maxidx = -1;" << std::endl;
    ss << "bottom_data += (n * channels + c) * height * width;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; h += dilation_h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; w += dilation_w) {" << std::endl;
    ss << "if (bottom_data[h * width + w] > maxval) {" << std::endl;
    ss << "maxidx = h * width + w;" << std::endl;
    ss << "maxval = bottom_data[maxidx];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = maxval;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "mask[index] = maxidx;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "top_mask[index] = maxidx;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AvePoolForwardSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "int_tp pool_size = 0;" << std::endl;
    ss << "int_tp pw = index % pooled_width;" << std::endl;
    ss << "int_tp ph = (index / pooled_width) % pooled_height;" << std::endl;
    ss << "int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "int_tp hend = hstart + ext_kernel_h;" << std::endl;
    ss << "int_tp wend = wstart + ext_kernel_w;" << std::endl;
    // Overspill over the image + pad does
    // not contribute to pool size
    ss << "while (hend > height + pad_h) {" << std::endl;
    ss << "hend -= dilation_h;" << std::endl;
    ss << "}" << std::endl;
    ss << "while (wend > width + pad_w) {" << std::endl;
    ss << "wend -= dilation_w;" << std::endl;
    ss << "}" << std::endl;
    ss << "Dtype aveval = 0;" << std::endl;
    ss << "bottom_data += (n * channels + c) * height * width;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; h += dilation_h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; w += dilation_w) {" << std::endl;
    ss << "if (h >= 0 && h < height && w >= 0 && w < width) {" << std::endl;
    ss << "aveval += bottom_data[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "++pool_size;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = aveval / ((Dtype)pool_size);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "rand_idx", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("StoPoolForwardTrainSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "int_tp pw = index % pooled_width;" << std::endl;
    ss << "int_tp ph = (index / pooled_width) % pooled_height;" << std::endl;
    ss << "int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h;" << std::endl;
    ss << "int_tp hend = min((int_tpc) (hstart + ext_kernel_h),"
        " (int_tpc) height);" << std::endl;
    ss << "int_tp wstart = pw * stride_w;" << std::endl;
    ss << "int_tp wend = min((int_tpc) (wstart + ext_kernel_w),"
       << " (int_tpc) width);" << std::endl;
    ss << "Dtype cumsum = 0.;" << std::endl;
    ss << "bottom_data += (n * channels + c) * height * width;" << std::endl;
    // First pass: get sum
    ss << "for (int_tp h = hstart; h < hend; h += dilation_h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; w += dilation_w) {" << std::endl;
    ss << "cumsum += bottom_data[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "Dtype thres = rand_idx[index] * cumsum;" << std::endl;
    // Second pass: get value, and set index.
    ss << "cumsum = 0;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; h += dilation_h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; w += dilation_w) {" << std::endl;
    ss << "cumsum += bottom_data[h * width + w];" << std::endl;
    ss << "if (cumsum >= thres) {" << std::endl;
    ss << "rand_idx[index] = ((n * channels + c) * height + h) * width + w;"
       << std::endl;
    ss << "top_data[index] = bottom_data[h * width + w];" << std::endl;
    ss << "return;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("StoPoolForwardTestSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "int_tp pw = index % pooled_width;" << std::endl;
    ss << "int_tp ph = (index / pooled_width) % pooled_height;" << std::endl;
    ss << "int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h;" << std::endl;
    ss << "int_tp hend = min((int_tpc) (hstart + ext_kernel_h),"
       << " (int_tpc) height);" << std::endl;
    ss << "int_tp wstart = pw * stride_w;" << std::endl;
    ss << "int_tp wend = min((int_tpc) (wstart + ext_kernel_w),"
       << " (int_tpc) width);" << std::endl;
    // We set cumsum to be 0 to avoid divide-by-zero problems
    ss << "Dtype cumsum = DTYPE_MIN;" << std::endl;
    ss << "Dtype cumvalues = 0.;" << std::endl;
    ss << "bottom_data += (n * channels + c) * height * width;" << std::endl;
    // First pass: get sum
    ss << "for (int_tp h = hstart; h < hend; h += dilation_h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; w += dilation_w) {" << std::endl;
    ss << "cumsum += bottom_data[h * width + w];" << std::endl;
    ss << "cumvalues += bottom_data[h * width + w]"
       << " * bottom_data[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = cumvalues / cumsum;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolBackwardSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local index
    // find out the local offset
    ss << "int_tp w = index % width;" << std::endl;
    ss << "int_tp h = (index / width) % height;" << std::endl;
    ss << "int_tp c = (index / width / height) % channels;" << std::endl;
    ss << "int_tp n = index / width / height / channels;" << std::endl;
    ss << "int_tp phstart ="
       << " (h + pad_h < ext_kernel_h) ? 0 : (h + pad_h - ext_kernel_h)"
       << " / stride_h + 1;" << std::endl;
    ss << "int_tp phend = min((int_tpc) ((h + pad_h) / stride_h + 1L),"
       << " (int_tpc) pooled_height);" << std::endl;
    ss << "int_tp pwstart ="
       << " (w + pad_w < ext_kernel_w) ? 0 : (w + pad_w - ext_kernel_w)"
       << " / stride_w + 1;" << std::endl;
    ss << "int_tp pwend = min((int_tpc) ((w + pad_w) / stride_w + 1L),"
          " (int_tpc) pooled_width);" << std::endl;
    ss << "Dtype gradient = 0.0;" << std::endl;
    ss << "int_tp offset = (n * channels + c) * pooled_height * pooled_width;"
       << std::endl;
    ss << "top_diff += offset;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "mask += offset;" << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    ss << "if (mask[ph * pooled_width + pw] == h * width + w) {" << std::endl;
    ss << "gradient += top_diff[ph * pooled_width + pw];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "} else {" << std::endl;
    ss << "top_mask += offset;" << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    ss << "if (top_mask[ph * pooled_width + pw] == (Dtype)(h * width + w)) {"
       << std::endl;
    ss << "gradient += top_diff[ph * pooled_width + pw];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AvePoolBackwardSK", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local index
    // find out the local offset
    ss << "const int_tp w = index % width;" << std::endl;
    ss << "const int_tp h = (index / width) % height;" << std::endl;
    ss << "const int_tp c = (index / width / height) % channels;" << std::endl;
    ss << "const int_tp n = index / width / height / channels;" << std::endl;
    ss << "int_tp phstart = "
       << "(h + pad_h < ext_kernel_h) ? 0 :"
       << "(h + pad_h - ext_kernel_h) / stride_h + 1;" << std::endl;
    ss << "int_tp phend = min(((h + pad_h) / stride_h + 1), pooled_height);"
       << std::endl;
    ss << "int_tp pwstart = "
       << "(w + pad_w < ext_kernel_w) ? 0 :"
       << "(w + pad_w - ext_kernel_w) / stride_w + 1;" << std::endl;
    ss << "int_tp pwend = min(((w + pad_w) / stride_w + 1), pooled_width);"
       << std::endl;
    ss << "Dtype gradient = 0.0;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_diff_slice")
       << " = top_diff + (n * channels + c) * pooled_height * pooled_width;"
       << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    // figure out the pooling size
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "int_tp hend = min(hstart + ext_kernel_h, height + pad_h);"
       << std::endl;
    ss << "int_tp wend = min(wstart + ext_kernel_w, width + pad_w);"
       << std::endl;
    ss << "int_tp pool_size ="
       << "((hend - hstart - 1) / dilation_h + 1) *"
       << "((wend - wstart - 1) / dilation_w + 1);" << std::endl;
    ss << "if (h >= hstart && h < hend &&"
       << "(h - hstart) % dilation_h == 0 &&"
       << "w >= wstart && w < wend &&"
       << "(w - wstart) % dilation_w == 0) {" << std::endl;
    ss << "gradient += top_diff_slice[ph * pooled_width + pw]"
       << " / ((Dtype)pool_size);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolForward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp pw = index % pooled_width;" << std::endl;
    ss << "const int_tp ph = (index / pooled_width) % pooled_height;"
       << std::endl;
    ss << "const int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "const int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "const int_tp hend = min((int_tpc) (hstart + kernel_h),"
       << " (int_tpc) height);" << std::endl;
    ss << "const int_tp wend = min((int_tpc) (wstart + kernel_w),"
       << " (int_tpc) width);" << std::endl;
    ss << "hstart = max((int_tpc) (hstart), (int_tpc) (0));" << std::endl;
    ss << "wstart = max((int_tpc) (wstart), (int_tpc) (0));" << std::endl;
    ss << "Dtype maxval = -DTYPE_MAX;" << std::endl;
    ss << "int_tp maxidx = -1;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "bottom_slice")
       << " = bottom_data + (n * channels + c) * height * width;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; ++h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; ++w) {" << std::endl;
    ss << "if (bottom_slice[h * width + w] > maxval) {" << std::endl;
    ss << "maxidx = h * width + w;" << std::endl;
    ss << "maxval = bottom_slice[maxidx];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = maxval;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "mask[index] = maxidx;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "top_mask[index] = maxidx;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AvePoolForward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp pw = index % pooled_width;" << std::endl;
    ss << "const int_tp ph = (index / pooled_width) % pooled_height;"
       << std::endl;
    ss << "const int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "const int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "int_tp hend = min((int_tpc) (hstart + kernel_h),"
       << " (int_tpc) (height + pad_h));" << std::endl;
    ss << "int_tp wend = min((int_tpc) (wstart + kernel_w),"
       << " (int_tpc) (width + pad_w));" << std::endl;
    ss << "const int_tp pool_size = (hend - hstart) * (wend - wstart);"
       << std::endl;
    ss << "hstart = max((int_tpc) (hstart), (int_tpc) (0));" << std::endl;
    ss << "wstart = max((int_tpc) (wstart), (int_tpc) (0));" << std::endl;
    ss << "hend = min((int_tpc) (hend), (int_tpc) (height));" << std::endl;
    ss << "wend = min((int_tpc) (wend), (int_tpc) (width));" << std::endl;
    ss << "Dtype aveval = 0;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "bottom_slice")
       << " = bottom_data + (n * channels + c) * height * width;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; ++h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; ++w) {" << std::endl;
    ss << "aveval += bottom_slice[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = aveval / ((Dtype)pool_size);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "rand_idx", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("StoPoolForwardTrain", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp pw = index % pooled_width;" << std::endl;
    ss << "const int_tp ph = (index / pooled_width) % pooled_height;"
       << std::endl;
    ss << "const int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "const int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "const int_tp hstart = ph * stride_h;" << std::endl;
    ss << "const int_tp hend = min((int_tpc) (hstart + kernel_h),"
       << " (int_tpc) height);" << std::endl;
    ss << "const int_tp wstart = pw * stride_w;" << std::endl;
    ss << "const int_tp wend = min((int_tpc) (wstart + kernel_w),"
       << " (int_tpc) width);" << std::endl;
    ss << "Dtype cumsum = 0.;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "bottom_slice")
       << " = bottom_data + (n * channels + c) * height * width;" << std::endl;
    // First pass: get sum
    ss << "for (int_tp h = hstart; h < hend; ++h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; ++w) {" << std::endl;
    ss << "cumsum += bottom_slice[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "const float thres = rand_idx[index] * cumsum;" << std::endl;
    // Second pass: get value, and set index.
    ss << "cumsum = 0;" << std::endl;
    ss << "for (int_tp h = hstart; h < hend; ++h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; ++w) {" << std::endl;
    ss << "cumsum += bottom_slice[h * width + w];" << std::endl;
    ss << "if (cumsum >= ((Dtype)thres)) {" << std::endl;
    ss << "rand_idx[index] = ((n * channels + c) * height + h) * width + w;"
       << std::endl;
    ss << "top_data[index] = bottom_slice[h * width + w];" << std::endl;
    ss << "return;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("StoPoolForwardTest", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp pw = index % pooled_width;" << std::endl;
    ss << "const int_tp ph = (index / pooled_width) % pooled_height;"
       << std::endl;
    ss << "const int_tp c = (index / pooled_width / pooled_height) % channels;"
       << std::endl;
    ss << "const int_tp n = index / pooled_width / pooled_height / channels;"
       << std::endl;
    ss << "const int_tp hstart = ph * stride_h;" << std::endl;
    ss << "const int_tp hend = min((int_tpc) (hstart + kernel_h),"
       << " (int_tpc) height);" << std::endl;
    ss << "const int_tp wstart = pw * stride_w;" << std::endl;
    ss << "const int_tp wend = min((int_tpc) (wstart + kernel_w),"
       << " (int_tpc) width);" << std::endl;
    // We set cumsum to be 0 to avoid divide-by-zero problems
    ss << "Dtype cumsum = 0.;" << std::endl;
    ss << "Dtype cumvalues = 0.;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "bottom_slice")
       << " = bottom_data + (n * channels + c) * height * width;" << std::endl;
    // First pass: get sum
    ss << "for (int_tp h = hstart; h < hend; ++h) {" << std::endl;
    ss << "for (int_tp w = wstart; w < wend; ++w) {" << std::endl;
    ss << "cumsum += bottom_slice[h * width + w];" << std::endl;
    ss << "cumvalues += bottom_slice[h * width + w]"
       << " * bottom_slice[h * width + w];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "top_data[index] = (cumsum > (Dtype)(0.0)) ? "
       << "cumvalues / cumsum : (Dtype)(0.0);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local index
    // find out the local offset
    ss << "const int_tp w = index % width;" << std::endl;
    ss << "const int_tp h = (index / width) % height;" << std::endl;
    ss << "const int_tp c = (index / width / height) % channels;" << std::endl;
    ss << "const int_tp n = index / width / height / channels;" << std::endl;
    ss << "const int_tp phstart ="
       << "(h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;"
       << std::endl;
    ss << "const int_tp phend = min((int_tpc) ((h + pad_h) / stride_h + 1L),"
       << "(int_tpc) pooled_height);" << std::endl;
    ss << "const int_tp pwstart ="
       << "(w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;"
       << std::endl;
    ss << "const int_tp pwend = min((int_tpc) ((w + pad_w) / stride_w + 1L),"
       << " (int_tpc) pooled_width);" << std::endl;
    ss << "Dtype gradient = 0;" << std::endl;
    ss << "const int_tp offset = (n * channels + c)"
       << " * pooled_height * pooled_width;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_diff_slice")
       << " = top_diff + offset;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << this->device_program_->global_ptr("const int_tp", "mask_slice")
       << " = mask + offset;" << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    ss << "if (mask_slice[ph * pooled_width + pw] == h * width + w) {"
       << std::endl;
    ss << "gradient += top_diff_slice[ph * pooled_width + pw];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "} else {" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_mask_slice")
       << " = top_mask + offset;" << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    ss << "if (top_mask_slice[ph * pooled_width + pw] == "
       << "(Dtype)(h * width + w)) {" << std::endl;
    ss << "gradient += top_diff_slice[ph * pooled_width + pw];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AvePoolBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local index
    // find out the local offset
    ss << "const int_tp w = index % width + pad_w;" << std::endl;
    ss << "const int_tp h = (index / width) % height + pad_h;" << std::endl;
    ss << "const int_tp c = (index / width / height) % channels;" << std::endl;
    ss << "const int_tp n = index / width / height / channels;" << std::endl;
    ss << "const int_tp phstart = (h < kernel_h) ? 0"
       << " : (h - kernel_h) / stride_h + 1;" << std::endl;
    ss << "const int_tp phend = min((int_tpc) (h / stride_h + 1),"
       << " (int_tpc) (pooled_height));" << std::endl;
    ss << "const int_tp pwstart = (w < kernel_w) ? 0"
       << " : (w - kernel_w) / stride_w + 1;" << std::endl;
    ss << "const int_tp pwend = min((int_tpc) (w / stride_w + 1),"
       << " (int_tpc) (pooled_width));" << std::endl;
    ss << "Dtype gradient = 0;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_diff_slice")
       << " = top_diff + (n * channels + c) * pooled_height * pooled_width;"
       << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    // figure out the pooling size
    ss << "int_tp hstart = ph * stride_h - pad_h;" << std::endl;
    ss << "int_tp wstart = pw * stride_w - pad_w;" << std::endl;
    ss << "int_tp hend = min((int_tpc) (hstart + kernel_h),"
       << " (int_tpc) (height + pad_h));" << std::endl;
    ss << "int_tp wend = min((int_tpc) (wstart + kernel_w),"
       << " (int_tpc) (width + pad_w));" << std::endl;
    ss << "int_tp pool_size = (hend - hstart) * (wend - wstart);" << std::endl;
    ss << "gradient += top_diff_slice[ph * pooled_width + pw]"
       << " / ((Dtype)pool_size);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "rand_idx", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_h", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride_w", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("StoPoolBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local index
    // find out the local offset
    ss << "const int_tp w = index % width;" << std::endl;
    ss << "const int_tp h = (index / width) % height;" << std::endl;
    ss << "const int_tp c = (index / width / height) % channels;" << std::endl;
    ss << "const int_tp n = index / width / height / channels;" << std::endl;
    ss << "const int_tp phstart = (h < kernel_h) ? 0"
       << " : (h - kernel_h) / stride_h + 1;" << std::endl;
    ss << "const int_tp phend = min((int_tpc) (h / stride_h + 1),"
       << " (int_tpc) pooled_height);" << std::endl;
    ss << "const int_tp pwstart = (w < kernel_w) ? 0"
       << " : (w - kernel_w) / stride_w + 1;" << std::endl;
    ss << "const int_tp pwend = min((int_tpc) (w / stride_w + 1),"
       << " (int_tpc) pooled_width);" << std::endl;
    ss << "Dtype gradient = 0;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "rand_idx_slice")
       << " = rand_idx + (n * channels + c) * pooled_height * pooled_width;"
       << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_diff_slice")
       << " = top_diff + (n * channels + c) * pooled_height * pooled_width;"
       << std::endl;
    ss << "for (int_tp ph = phstart; ph < phend; ++ph) {" << std::endl;
    ss << "for (int_tp pw = pwstart; pw < pwend; ++pw) {" << std::endl;
    ss << "gradient += top_diff_slice[ph * pooled_width + pw]"
       << " * (index  == (int_tpc)(rand_idx_slice[ph * pooled_width + pw]) ?"
       << " (Dtype)1.0 : (Dtype)0.0);"  << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "n", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num_axes", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_size", KERNEL_ARG_CONST |
                                         KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_data", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolForwardND", args);
    ss << "int_tp d_idx[6];" << std::endl;
    ss << "int_tp d_start[6];" << std::endl;
    ss << "int_tp d_end[6];" << std::endl;
    ss << "int_tp d_iter[6];" << std::endl;
    ss << "int_tp i;" << std::endl;
    ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
    ss << "int_tp offset = 1;" << std::endl;
    ss << "int_tp num = index;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "d_idx[i] = num % pooled_size[i];" << std::endl;
    ss << "d_start[i] = d_idx[i] * stride[i] - pad[i];" << std::endl;
    ss << "d_end[i] = min((int_tpc) (d_start[i] + ext_kernel_size[i]),"
       << " (int_tpc) (size[i]));" << std::endl;
    ss << "while (d_start[i] < 0) {" << std::endl;
    ss << "d_start[i] += dilation[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "num /= pooled_size[i];" << std::endl;
    ss << "offset *= size[i];" << std::endl;
    ss << "d_iter[i] = d_start[i];" << std::endl;
    ss << "if (d_start[i] >= d_end[i]) {" << std::endl;
    ss << "top_data[index] = -DTYPE_MAX;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "mask[index] = -1;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "top_mask[index] = -1;" << std::endl;
    ss << "}" << std::endl;
    ss << "return;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "int_tp chan = num % channels;" << std::endl;
    ss << "num /= channels;" << std::endl;
    ss << "offset *= (num * channels + chan);" << std::endl;
    ss << "Dtype maxval = -DTYPE_MAX;" << std::endl;
    ss << "int_tp maxidx = -1;" << std::endl;
    ss << "int_tp final_offset = 0;" << std::endl;
    ss << "bool incremented;" << std::endl;
    ss << "do {" << std::endl;
    ss << "final_offset = 0;" << std::endl;
    ss << "int_tp size_prod = 1;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "final_offset += d_iter[i] * size_prod;" << std::endl;
    ss << "size_prod *= size[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (bottom_data[final_offset + offset] > maxval) {" << std::endl;
    ss << "maxidx = final_offset;" << std::endl;
    ss << "maxval = bottom_data[offset + final_offset];" << std::endl;
    ss << "}" << std::endl;
    ss << "incremented = false;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "if (d_iter[i] >= d_end[i] - dilation[i]) {" << std::endl;
    ss << "d_iter[i] = d_start[i];" << std::endl;
    ss << "} else {" << std::endl;
    ss << "d_iter[i] += dilation[i];" << std::endl;
    ss << "incremented = true;" << std::endl;
    ss << "break;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "} while (incremented);" << std::endl;
    ss << "top_data[index] = maxval;" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "mask[index] = maxidx;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "top_mask[index] = maxidx;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "n", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num_axes", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "top_mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pooled_size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "kernel_size", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ext_kernel_size", KERNEL_ARG_CONST |
                                         KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "stride", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dilation", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "pad", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("MaxPoolBackwardND", args);
    ss << "int_tp d_idx[6];" << std::endl;
    ss << "int_tp d_start[6];" << std::endl;
    ss << "int_tp d_end[6];" << std::endl;
    ss << "int_tp d_iter[6];" << std::endl;
    ss << "int_tp i;" << std::endl;

    ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
    // find out the local index
    // find out the local offset
    ss << "int_tp offset = 1;" << std::endl;
    ss << "int_tp num = index;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "d_idx[i] = num % size[i];" << std::endl;
    ss << "d_start[i] ="
       << " (d_idx[i] + pad[i] < ext_kernel_size[i]) ?"
       << " 0L : (d_idx[i] + pad[i] - ext_kernel_size[i]) / stride[i] + 1L;"
       << std::endl;
    ss << "d_end[i] = min((int_tpc) ((d_idx[i] + pad[i]) / stride[i]),"
       << " (int_tpc) (pooled_size[i] - 1L));" << std::endl;
    ss << "num /= size[i];" << std::endl;
    ss << "offset *= pooled_size[i];" << std::endl;
    ss << "d_iter[i] = d_start[i];" << std::endl;
    ss << "if (d_start[i] > d_end[i]) {" << std::endl;
    ss << "bottom_diff[index] = 0;" << std::endl;
    ss << "return;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "int_tp chan = num % channels;" << std::endl;
    ss << "num /= channels;" << std::endl;
    ss << "offset *= (num * channels + chan);" << std::endl;
    ss << "Dtype gradient = 0.0;" << std::endl;
    ss << "int_tp final_offset = 0;" << std::endl;
    ss << "int_tp im_offset = 0;" << std::endl;
    ss << "bool incremented;" << std::endl;
    ss << "do {" << std::endl;
    ss << "final_offset = offset;" << std::endl;
    ss << "im_offset = 0;" << std::endl;
    ss << "int_tp size_prod = 1;" << std::endl;
    ss << "int_tp pooled_size_prod = 1;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "final_offset += d_iter[i] * pooled_size_prod;" << std::endl;
    ss << "im_offset += d_idx[i] * size_prod;" << std::endl;
    ss << "size_prod *= size[i];" << std::endl;
    ss << "pooled_size_prod *= pooled_size[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (mask) {" << std::endl;
    ss << "if (mask[final_offset] == im_offset) {" << std::endl;
    ss << "gradient += top_diff[final_offset];" << std::endl;
    ss << "}" << std::endl;
    ss << "} else {" << std::endl;
    ss << "if (top_mask[final_offset] == (Dtype)im_offset) {" << std::endl;
    ss << "gradient += top_diff[final_offset];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "incremented = false;" << std::endl;
    ss << "for (i = num_axes - 1; i >= 0; --i) {" << std::endl;
    ss << "if (d_iter[i] >= d_end[i]) {" << std::endl;
    ss << "d_iter[i] = d_start[i];" << std::endl;
    ss << "} else {" << std::endl;
    ss << "++d_iter[i];" << std::endl;
    ss << "incremented = true;" << std::endl;
    ss << "break;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "} while (incremented);" << std::endl;
    ss << "bottom_diff[index] = gradient;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}



template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  uint_tp count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  vptr<int_tp> mask;
  vptr<Dtype> top_mask;

  if (num_spatial_axes_ == 2) {
    int_tp kernel_h = kernel_shape_.cpu_data()[0];
    int_tp kernel_w = kernel_shape_.cpu_data()[1];
    int_tp stride_h = stride_.cpu_data()[0];
    int_tp stride_w = stride_.cpu_data()[1];
    int_tp pad_h = pad_.cpu_data()[0];
    int_tp pad_w = pad_.cpu_data()[1];
    int_tp dilation_h = dilation_.cpu_data()[0];
    int_tp dilation_w = dilation_.cpu_data()[1];
    int_tp num = bottom[0]->shape(0);
    int_tp height = size_.cpu_data()[0];
    int_tp width = size_.cpu_data()[1];
    int_tp pooled_height = pooled_size_.cpu_data()[0];
    int_tp pooled_width = pooled_size_.cpu_data()[1];
    int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
    int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[1];

    // 2D case
    if (use_skernel_) {
      // 2D-SK case
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX: {
          if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
          } else {
            mask = max_idx_.mutable_gpu_data();
          }

          shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("MaxPoolForwardSK");
          kernel->add_arg(&count);
          kernel->add_arg(&bottom_data);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&ext_kernel_h);
          kernel->add_arg(&ext_kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&dilation_h);
          kernel->add_arg(&dilation_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&top_data);
          kernel->add_arg(&mask);
          kernel->add_arg(&top_mask);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_AVE: {
          shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("AvePoolForwardSK");
          kernel->add_arg(&count);
          kernel->add_arg(&bottom_data);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&ext_kernel_h);
          kernel->add_arg(&ext_kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&dilation_h);
          kernel->add_arg(&dilation_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&top_data);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_STOCHASTIC: {
          if (this->phase_ == caffe::TRAIN) {
            // We need to create the random index as well.
            this->device_->template rng_uniform<Dtype>(count, Dtype(0),
                           Dtype(1), rand_idx_.mutable_gpu_data());

            shared_ptr<DeviceKernel> kernel =
                      this->device_program_->GetKernel("StoPoolForwardTrainSK");
            kernel->add_arg(&count);
            kernel->add_arg(&bottom_data);
            kernel->add_arg(&num);
            kernel->add_arg(&channels_);
            kernel->add_arg(&height);
            kernel->add_arg(&width);
            kernel->add_arg(&pooled_height);
            kernel->add_arg(&pooled_width);
            kernel->add_arg(&kernel_h);
            kernel->add_arg(&kernel_w);
            kernel->add_arg(&ext_kernel_h);
            kernel->add_arg(&ext_kernel_w);
            kernel->add_arg(&stride_h);
            kernel->add_arg(&stride_w);
            kernel->add_arg(&dilation_h);
            kernel->add_arg(&dilation_w);
            kernel->add_arg(&top_data);

            vector<size_t> work_size(1, count);
            vector<size_t> group;
            vector<size_t> local;
            this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                       true);
            kernel->Execute(group, local);
          } else {
            shared_ptr<DeviceKernel> kernel =
                       this->device_program_->GetKernel("StoPoolForwardTestSK");
            kernel->add_arg(&count);
            kernel->add_arg(&bottom_data);
            kernel->add_arg(&num);
            kernel->add_arg(&channels_);
            kernel->add_arg(&height);
            kernel->add_arg(&width);
            kernel->add_arg(&pooled_height);
            kernel->add_arg(&pooled_width);
            kernel->add_arg(&kernel_h);
            kernel->add_arg(&kernel_w);
            kernel->add_arg(&ext_kernel_h);
            kernel->add_arg(&ext_kernel_w);
            kernel->add_arg(&stride_h);
            kernel->add_arg(&stride_w);
            kernel->add_arg(&dilation_h);
            kernel->add_arg(&dilation_w);
            kernel->add_arg(&top_data);

            vector<size_t> work_size(1, count);
            vector<size_t> group;
            vector<size_t> local;
            this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                       true);
            kernel->Execute(group, local);
          }
          break;
        }
        default: {
          LOG(FATAL)<< "Unknown pooling method.";
        }
      }
    } else {
      // 2D case
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX: {
          if (use_top_mask) {
            top_mask = top[1]->mutable_gpu_data();
          } else {
            mask = max_idx_.mutable_gpu_data();
          }

          shared_ptr<DeviceKernel> kernel =
                             this->device_program_->GetKernel("MaxPoolForward");
          kernel->add_arg(&count);
          kernel->add_arg(&bottom_data);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&top_data);
          kernel->add_arg(&mask);
          kernel->add_arg(&top_mask);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_AVE: {
          shared_ptr<DeviceKernel> kernel =
                             this->device_program_->GetKernel("AvePoolForward");
          kernel->add_arg(&count);
          kernel->add_arg(&bottom_data);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&top_data);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_STOCHASTIC: {
          if (this->phase_ == TRAIN) {
            // We need to create the random index as well.
            this->device_->template rng_uniform<Dtype>(count, Dtype(0),
                           Dtype(1), rand_idx_.mutable_gpu_data());

            vptr<Dtype> rand_idx_data = rand_idx_.mutable_gpu_data();

            shared_ptr<DeviceKernel> kernel =
                        this->device_program_->GetKernel("StoPoolForwardTrain");
            kernel->add_arg(&count);
            kernel->add_arg(&bottom_data);
            kernel->add_arg(&num);
            kernel->add_arg(&channels_);
            kernel->add_arg(&height);
            kernel->add_arg(&width);
            kernel->add_arg(&pooled_height);
            kernel->add_arg(&pooled_width);
            kernel->add_arg(&kernel_h);
            kernel->add_arg(&kernel_w);
            kernel->add_arg(&stride_h);
            kernel->add_arg(&stride_w);
            kernel->add_arg(&rand_idx_data);
            kernel->add_arg(&top_data);

            vector<size_t> work_size(1, count);
            vector<size_t> group;
            vector<size_t> local;
            this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                       true);
            kernel->Execute(group, local);
          } else {
            shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("StoPoolForwardTest");
            kernel->add_arg(&count);
            kernel->add_arg(&bottom_data);
            kernel->add_arg(&num);
            kernel->add_arg(&channels_);
            kernel->add_arg(&height);
            kernel->add_arg(&width);
            kernel->add_arg(&pooled_height);
            kernel->add_arg(&pooled_width);
            kernel->add_arg(&kernel_h);
            kernel->add_arg(&kernel_w);
            kernel->add_arg(&stride_h);
            kernel->add_arg(&stride_w);
            kernel->add_arg(&top_data);

            vector<size_t> work_size(1, count);
            vector<size_t> group;
            vector<size_t> local;
            this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                       true);
            kernel->Execute(group, local);
          }
          break;
        }
        default: {
          LOG(FATAL)<< "Unknown pooling method.";
        }
      }
    }
  } else {
    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX: {
        if (use_top_mask) {
          top_mask = top[1]->mutable_gpu_data();
        } else {
          mask = max_idx_.mutable_gpu_data();
        }

        vptr<const int_tp> size_data = size_.gpu_data();
        vptr<const int_tp> pooled_size_data = pooled_size_.gpu_data();
        vptr<const int_tp> kernel_shape_data = kernel_shape_.gpu_data();
        vptr<const int_tp> ext_kernel_shape_data = ext_kernel_shape_.gpu_data();
        vptr<const int_tp> stride_data = stride_.gpu_data();
        vptr<const int_tp> dilation_data = dilation_.gpu_data();
        vptr<const int_tp> pad_data = pad_.gpu_data();

        shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("MaxPoolForwardND");
        kernel->add_arg(&count);
        kernel->add_arg(&num_spatial_axes_);
        kernel->add_arg(&bottom_data);
        kernel->add_arg(&channels_);
        kernel->add_arg(&size_data);
        kernel->add_arg(&pooled_size_data);
        kernel->add_arg(&kernel_shape_data);
        kernel->add_arg(&ext_kernel_shape_data);
        kernel->add_arg(&stride_data);
        kernel->add_arg(&dilation_data);
        kernel->add_arg(&pad_data);
        kernel->add_arg(&top_data);
        kernel->add_arg(&mask);
        kernel->add_arg(&top_mask);

        vector<size_t> work_size(1, count);
        vector<size_t> group;
        vector<size_t> local;
        this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                   true);
        kernel->Execute(group, local);
        break;
      }
      default: {
        LOG(FATAL)<< "Unknown pooling method.";
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp count = bottom[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  vptr<const int_tp> mask;
  vptr<const Dtype> top_mask;

  this->device_->set(count, Dtype(0.), bottom_diff);

  if (num_spatial_axes_ == 2) {
    int_tp kernel_h = kernel_shape_.cpu_data()[0];
    int_tp kernel_w = kernel_shape_.cpu_data()[1];
    int_tp stride_h = stride_.cpu_data()[0];
    int_tp stride_w = stride_.cpu_data()[1];
    int_tp pad_h = pad_.cpu_data()[0];
    int_tp pad_w = pad_.cpu_data()[1];
    int_tp dilation_h = dilation_.cpu_data()[0];
    int_tp dilation_w = dilation_.cpu_data()[1];
    int_tp num = top[0]->shape(0);
    int_tp height = size_.cpu_data()[0];
    int_tp width = size_.cpu_data()[1];
    int_tp pooled_height = pooled_size_.cpu_data()[0];
    int_tp pooled_width = pooled_size_.cpu_data()[1];
    int_tp ext_kernel_h = ext_kernel_shape_.cpu_data()[0];
    int_tp ext_kernel_w = ext_kernel_shape_.cpu_data()[1];

    if (use_skernel_) {
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX: {
          if (use_top_mask) {
            top_mask = top[1]->gpu_data();
          } else {
            mask = max_idx_.gpu_data();
          }

          shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("MaxPoolBackwardSK");
          kernel->add_arg(&count);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&mask);
          kernel->add_arg(&top_mask);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&ext_kernel_h);
          kernel->add_arg(&ext_kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&dilation_h);
          kernel->add_arg(&dilation_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_AVE: {
          shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("AvePoolBackwardSK");
          kernel->add_arg(&count);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&ext_kernel_h);
          kernel->add_arg(&ext_kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&dilation_h);
          kernel->add_arg(&dilation_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        default: {
          LOG(FATAL)<<
          "Unknown or unsupported pooling method in Backward_gpu().";
        }
      }
    } else {
      switch (this->layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX: {
          if (use_top_mask) {
            top_mask = top[1]->gpu_data();
          } else {
            mask = max_idx_.gpu_data();
          }

          shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("MaxPoolBackward");
          kernel->add_arg(&count);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&mask);
          kernel->add_arg(&top_mask);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_AVE: {
          shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("AvePoolBackward");
          kernel->add_arg(&count);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&pad_h);
          kernel->add_arg(&pad_w);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
          kernel->Execute(group, local);
          break;
        }
        case PoolingParameter_PoolMethod_STOCHASTIC: {
          vptr<const Dtype> rand_idx_data = rand_idx_.gpu_data();

          shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("StoPoolBackward");
          kernel->add_arg(&count);
          kernel->add_arg(&rand_idx_data);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&num);
          kernel->add_arg(&channels_);
          kernel->add_arg(&height);
          kernel->add_arg(&width);
          kernel->add_arg(&pooled_height);
          kernel->add_arg(&pooled_width);
          kernel->add_arg(&kernel_h);
          kernel->add_arg(&kernel_w);
          kernel->add_arg(&stride_h);
          kernel->add_arg(&stride_w);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        default: {
          LOG(FATAL) << "Unknown pooling method.";
        }
      }
    }
  } else {
    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX: {
        if (use_top_mask) {
          top_mask = top[1]->gpu_data();
        } else {
          mask = max_idx_.gpu_data();
        }

        vptr<const int_tp> size_data = size_.gpu_data();
        vptr<const int_tp> pooled_size_data = pooled_size_.gpu_data();
        vptr<const int_tp> kernel_shape_data = kernel_shape_.gpu_data();
        vptr<const int_tp> ext_kernel_shape_data = ext_kernel_shape_.gpu_data();
        vptr<const int_tp> stride_data = stride_.gpu_data();
        vptr<const int_tp> dilation_data = dilation_.gpu_data();
        vptr<const int_tp> pad_data = pad_.gpu_data();

        shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("MaxPoolBackwardND");
        kernel->add_arg(&count);
        kernel->add_arg(&num_spatial_axes_);
        kernel->add_arg(&top_diff);
        kernel->add_arg(&mask);
        kernel->add_arg(&top_mask);
        kernel->add_arg(&channels_);
        kernel->add_arg(&size_data);
        kernel->add_arg(&pooled_size_data);
        kernel->add_arg(&kernel_shape_data);
        kernel->add_arg(&ext_kernel_shape_data);
        kernel->add_arg(&stride_data);
        kernel->add_arg(&dilation_data);
        kernel->add_arg(&pad_data);
        kernel->add_arg(&bottom_diff);

        vector<size_t> work_size(1, count);
        vector<size_t> group;
        vector<size_t> local;
        this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                   true);
        kernel->Execute(group, local);
        break;
      }
      default: {
        LOG(FATAL) <<
        "Unknown or unsupported pooling method in Backward_gpu().";
      }
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (float), (float), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (double), (double),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (uint8_t), (uint8_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (uint16_t), (uint16_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (uint32_t), (uint32_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, GenerateProgram,
                                  (uint64_t), (uint64_t),  PROTO_TYPES);

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (float), (float), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (double), (double), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (uint8_t), (uint8_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (uint16_t), (uint16_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (uint32_t), (uint32_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Forward_gpu,
                                  (uint64_t), (uint64_t),  PROTO_TYPES);

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (half_fp), (half_fp), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (float), (float), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (double), (double), PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (uint8_t), (uint8_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (uint16_t), (uint16_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (uint32_t), (uint32_t),  PROTO_TYPES);
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PoolingLayer, Backward_gpu,
                                  (uint64_t), (uint64_t),  PROTO_TYPES);

}  // namespace caffe
