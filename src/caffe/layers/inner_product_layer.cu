#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#ifdef USE_GREENTEA
#include "viennacl/tools/sha1.hpp"
#include "caffe/util/benchmark.hpp"
#endif

namespace caffe {

struct gemm_callback_arg {
  std::vector<cl_event> evs;
  std::vector<cl_mem> imgs;
};

static void CL_CALLBACK gemm_callback (cl_event event,
                                cl_int event_command_exec_status,
                                void *user_data) {
  struct gemm_callback_arg *arg = (struct gemm_callback_arg *) user_data;
  for(int i = 0; i < arg->evs.size(); i++) {
    clReleaseEvent(arg->evs[i]);
  }

  for(int i = 0; i < arg->imgs.size(); i++) {
    clReleaseMemObject(arg->imgs[i]);
  }
  delete arg;
}

// Create and copy buffer to image for GEMM's matrix A and B.
// Will return image to caller if the input image is NULL. Otherwise,
// will use the image directly. It's caller's responsibility to
// release the created image.
template<typename Dtype>
static void greentea_gpu_gemm_copy_buffer_to_image(int_tp ctx_id,
                 cl_mem *image, cl_mem buffer, int offset,
                 bool is_matrix_a, bool transpose,
                 bool padding, int padded_height,
                 int padded_width, int height,
                 int width,  int ld, int wait_list_size,
                 cl_event *wait_list,
                 cl_event *event) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
  ->program();
  cl_image_desc desc;
  cl_image_format format;

  bool halfPrecisionMode = !std::is_same<Dtype, float>::value;

  memset(&desc, 0, sizeof(desc));
  int src_offset = halfPrecisionMode ? sizeof(unsigned short) * offset : sizeof(float) * offset;
  if (!is_matrix_a && transpose) {
  // For matrix B with transpose, we need to handle them differently.
  // As we can't use the sub group block read to get a row easily,
  // we have to use CL_FLOAT type with read_imagef to get the row.
    cl_int err;
    if(halfPrecisionMode) {
      format.image_channel_data_type = CL_HALF_FLOAT;
    } else {
      format.image_channel_data_type = CL_FLOAT;
    }
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    format.image_channel_order = CL_R;

    desc.image_height = height;
    if (*image == NULL) {
      *image = clCreateImage(
                            ctx.handle().get(),
                            CL_MEM_READ_WRITE,
                            &format,
                            &desc,
                            NULL,
                            &err);
      OCL_CHECK(err);
    }

    if(ld == width) {
      size_t origin[] = {0, 0, 0};
      size_t region[] = {(size_t)desc.image_width,
                         (size_t)desc.image_height, 1};

      OCL_CHECK(clEnqueueCopyBufferToImage(ctx.get_queue().handle().get(),
                                 buffer, *image, src_offset,
                                 origin, region, wait_list_size,
                                 wait_list, event));
    } else {
      viennacl::ocl::kernel &oclk_gemm_copy = program.get_kernel(
        CL_KERNEL_SELECT("gemm_buffer_copy_image_transpose"));

      size_t global_copy[2];
      global_copy[0] = width;
      global_copy[1] = height;
      oclk_gemm_copy.arg(0, WrapHandle(buffer, &ctx));
      oclk_gemm_copy.arg(1, WrapHandle(*image, &ctx));
      oclk_gemm_copy.arg(2, offset);
      oclk_gemm_copy.arg(3, width);
      oclk_gemm_copy.arg(4, height);
      oclk_gemm_copy.arg(5, ld);
      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_gemm_copy.handle().get(),
                                       2, NULL, global_copy, NULL,
                                       wait_list_size, wait_list,
                                       event));
    }
  } else {
    if (*image == NULL) {
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      if(halfPrecisionMode) {
        format.image_channel_data_type = CL_HALF_FLOAT;
        format.image_channel_order = CL_R;
      } else {
        format.image_channel_data_type = CL_UNSIGNED_INT8;
        format.image_channel_order = CL_RGBA;
      }

      if (!padding) {
        desc.image_width = width;
        desc.image_height = height;
      } else {
        desc.image_width = padded_width;
        desc.image_height = padded_height;
      }
      cl_int err;
        *image = clCreateImage(ctx.handle().get(),
                            desc.buffer ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE,
                            &format,
                            &desc,
                            NULL,
                            &err);
        OCL_CHECK(err);
    }
    if (!padding && desc.buffer != NULL)
      return;
    if (!padding && desc.buffer == NULL) {
    // copy without padding.
      size_t origin[] = {0, 0, 0};
      size_t region[] = {(size_t)width, (size_t)height, 1};
      OCL_CHECK(clEnqueueCopyBufferToImage(ctx.get_queue().handle().get(),
                                 buffer, *image, src_offset,
                                 origin, region, wait_list_size, wait_list, event));
    } else {
      viennacl::ocl::kernel &oclk_gemm_copy = program.get_kernel(
        CL_KERNEL_SELECT("gemm_buffer_copy_image_no_transpose"));

      size_t global_copy[2];
      global_copy[0] = padding ? padded_width : width;
      global_copy[1] = padding ? padded_height : height;
      oclk_gemm_copy.arg(0, WrapHandle(buffer, &ctx));
      oclk_gemm_copy.arg(1, WrapHandle(*image, &ctx));
      oclk_gemm_copy.arg(2, offset);
      oclk_gemm_copy.arg(3, width);
      oclk_gemm_copy.arg(4, height);
      oclk_gemm_copy.arg(5, ld);
      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_gemm_copy.handle().get(),
                                       2, NULL, global_copy, NULL,
                                       wait_list_size, wait_list,
                                       event));
    }
  }
}

template<typename Dtype>
static void greentea_gpu_fast_image_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC, bool is_image_a, bool is_image_b,
                       enum gemm_type_t gemm_type, const size_t max_image_size) {
  CHECK_EQ(gemm_type == GEMM_TYPE_FAST_IMAGE_32_1
           || gemm_type == GEMM_TYPE_FAST_IMAGE_32_2
           || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE, true)
    << "Invalid fast image gemm type." << std::endl;
  if (is_image_a)
    CHECK_EQ(offA, 0) << "Invalid input image offset." << std::endl;

  if (is_image_b)
    CHECK_EQ(offB, 0) << "Invalid input image offset." << std::endl;

  bool halfPrecisionMode = !std::is_same<Dtype, float>::value;
  int widthA = (TransA == CblasNoTrans) ? K : M;
  int heightA = (TransA == CblasNoTrans) ? M : K;
  int widthB = (TransB == CblasNoTrans) ? N : K;
  int heightB = (TransB == CblasNoTrans) ? K : N;

  int ldA = widthA;
  int ldB = widthB;
  int ldC = N;

  int A_start_x = 0, A_start_y = 0, B_start_x = 0, B_start_y = 0, C_start_x = 0, C_start_y = 0;
  int blocksize = 1024;
  if (gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
    blocksize = max_image_size;
  int blockA_width = blocksize;
  int blockA_height = blocksize;
  int blockB_width = blocksize;
  int blockB_height = blocksize;
  int blockC_width = blocksize;
  int blockC_height = blocksize;

  int use_buffer_indicator = halfPrecisionMode ? 16 : 8;
  // To fix the edge problem casued by the sub group block read.
  // we have to pad the image if it's not multiple of tile.
  // just padding one line is enough as the sub group block read
  // will clamp to edge according to the spec.

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
                                     ->program();

  cl_mem ImA = NULL;
  cl_mem ImB = NULL;

  viennacl::ocl::kernel *oclk_gemm_float;
  std::string kernel_name("gemm_");
  if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1
      || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
    kernel_name += "32_1_";
  else
    kernel_name += "32_2_";

  if (TransA == CblasNoTrans)
    kernel_name += "N";
  else
    kernel_name += "T";

  if (TransB == CblasNoTrans)
    kernel_name += "N_";
  else {
    kernel_name += "T_";
    if (is_image_b || (K % use_buffer_indicator != 0)) {
      kernel_name += "SCALAR_";
    } else {
      kernel_name += "BUFFER_";
    }
  }

  if (alpha == 1)
    kernel_name += "1_";
  else
    kernel_name += "0_";

  if (beta == 0)
    kernel_name += "0";
  else
    kernel_name += "1";

  if(halfPrecisionMode) {
    kernel_name += "_half";
  } else {
    kernel_name += "_float";
  }

  oclk_gemm_float = &program.get_kernel(kernel_name);
  while(C_start_y < M) {
    blockC_width = std::min((int)N - C_start_x, blocksize);
    blockC_height = std::min((int)M - C_start_y, blocksize);

    int isFirstColBlock = 1;
    for(int k = 0; k < K; k += blocksize) {
      cl_event ev[5];
      cl_uint ev_idx = 0;
      memset(ev, 0, sizeof(cl_event) * 5);
      struct gemm_callback_arg * arg = new gemm_callback_arg;

      blockA_width = std::min(widthA - A_start_x, blocksize);
      blockA_height = std::min(heightA - A_start_y, blocksize);
      blockB_width = std::min(widthB - B_start_x, blocksize);
      blockB_height = std::min(heightB - B_start_y, blocksize);
      int block_Ksize = std::min((int)K - k, blocksize);

      int padded_k = block_Ksize + ((block_Ksize & 7) ? (8 - (block_Ksize & 7)) : 0);
      int imageA_w = (TransA == CblasNoTrans) ? padded_k : blockA_width;
      int imageA_h = (TransA == CblasNoTrans) ? blockA_height : padded_k;
      int imageB_w = (TransB == CblasNoTrans) ? blockB_width : padded_k;
      int imageB_h = (TransB == CblasNoTrans) ? padded_k : blockB_height;

      int blockA_offset = offA + A_start_y * ldA + A_start_x;
      int blockB_offset = offB + B_start_y * ldB + B_start_x;
      int blockC_offset = offC + C_start_y * ldC + C_start_x;
      if (TransB == CblasNoTrans) {
        bool padding_A = false;
        bool padding_B = false;

        if(halfPrecisionMode && is_image_b) {
          padding_A = true;
        }

        if (!is_image_a && !is_image_b) {
          if (M * K < N * K)
            padding_B = true;
          else
            padding_A = true;
        }

        if (!is_image_a) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImA, A, blockA_offset,
                                    true, TransA != CblasNoTrans,
                                    padding_A, imageA_h, imageA_w,
                                    blockA_height, blockA_width, ldA, 0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
        if (!is_image_b) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImB, B, blockB_offset,
                                    false, false,
                                    padding_B, imageB_h, imageB_w,
                                    blockB_height, blockB_width, ldB, 0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
      } else {
        // We will use normal read_imagef to read image B when B has transpose.
        // thus we don't need to pad image A at all.
        if (!is_image_a) {
          bool padding;
          padding = !is_image_b || halfPrecisionMode;
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImA, A, blockA_offset,
                                    true, TransA != CblasNoTrans,
                                    padding, imageA_h, imageA_w,
                                    blockA_height, blockA_width, ldA, 0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }

        if(!is_image_b && (K % use_buffer_indicator != 0)) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImB, B, blockB_offset,
                                    false, true, false, imageB_h, imageB_w,
                                    blockB_height, blockB_width, ldB, 0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
      }
      if (is_image_a)
        ImA = A;
      if (is_image_b)
        ImB = B;

      size_t global[2];
      if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
          gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE ) {
        if(halfPrecisionMode) {
          global[0] = (size_t)( blockC_width + 15 ) & ~15;
        } else {
          global[0] = (size_t)( blockC_width + 7 ) & ~7;
        }
      }
      else {
        if(halfPrecisionMode) {
          global[0] = (size_t)( (blockC_width / 2 ) + 15 ) ^ ~15;
        } else {
          global[0] = (size_t)( (blockC_width / 2 ) + 7 ) ^ ~7;
        }
      }
      global[1]  = (size_t)(blockC_height + 31) / 32;

      size_t local[2];

      if (halfPrecisionMode) {
        local[0] = 16;
      } else {
        local[0] = 8;
      }
      local[1] = 1;

      cl_uint arg_idx = 0;
      oclk_gemm_float->arg(arg_idx++, WrapHandle(ImA, &ctx));
      if (TransB == CblasNoTrans || is_image_b || (K % use_buffer_indicator != 0))
        oclk_gemm_float->arg(arg_idx++, WrapHandle(ImB, &ctx));
      else {
        oclk_gemm_float->arg(arg_idx++, WrapHandle(B, &ctx));
        oclk_gemm_float->arg(arg_idx++, blockB_offset);
        oclk_gemm_float->arg(arg_idx++, ldB);
      }
      oclk_gemm_float->arg(arg_idx++, WrapHandle(C, &ctx));
      oclk_gemm_float->arg(arg_idx++, blockC_offset);
      oclk_gemm_float->arg(arg_idx++, blockC_height);
      oclk_gemm_float->arg(arg_idx++, blockC_width);
      oclk_gemm_float->arg(arg_idx++, ldC);
      oclk_gemm_float->arg(arg_idx++, fixup_arg_type(alpha));
      oclk_gemm_float->arg(arg_idx++, fixup_arg_type(beta));
      oclk_gemm_float->arg(arg_idx++, padded_k);
      if (TransB != CblasNoTrans)
        oclk_gemm_float->arg(arg_idx++, block_Ksize);
      oclk_gemm_float->arg(arg_idx++, isFirstColBlock);

      cl_event *wait_list = NULL;
      if (ev_idx != 0)
        wait_list = &ev[0];
      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_gemm_float->handle().get(), 2, NULL,
                                       global, local, ev_idx,
                                       wait_list, &ev[ev_idx]));
      if(TransA == CblasNoTrans)
        A_start_x += blockA_width;
      else
        A_start_y += blockA_height;

      if(TransB == CblasNoTrans)
        B_start_y += blockB_height;
      else
        B_start_x += blockB_width;

      isFirstColBlock = 0;
      arg->evs.assign(ev, ev + ev_idx + 1);
      clSetEventCallback(ev[ev_idx], CL_COMPLETE, &gemm_callback, (void*)arg);
    }

    C_start_x += blockC_width;
    if(TransA == CblasNoTrans)
      A_start_x = 0;
    else
      A_start_y = 0;
    if(TransB == CblasNoTrans) {
      B_start_x += blockB_width;
      B_start_y = 0;
    } else {
      B_start_y += blockB_height;
      B_start_x = 0;
    }
    if(C_start_x >= N) {
      C_start_x = 0;
      B_start_x = 0;
      B_start_y = 0;
      C_start_y += blockC_height;
      if(TransA == CblasNoTrans)
        A_start_y += blockA_height;
      else
        A_start_x += blockA_width;
    }
  }

  if(ImA && !is_image_a)
    clReleaseMemObject(ImA);
  if(ImB && !is_image_b)
    clReleaseMemObject(ImB);
}

template<typename Dtype>
static void greentea_gpu_fast_buffer_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC, enum gemm_type_t gemm_type) {
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_BUFFER, true)
      << "Invalid fast buffer gemm type." << std::endl;

    cl_event ev;

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
    viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
                                       ->program();
    bool halfPrecisionMode = !std::is_same<Dtype, float>::value;

    size_t sub_group_size = 8;
    bool is_small_batch = (M == 2 || M == 4 || M == 8);
    viennacl::ocl::kernel *oclk_gemm_float;
    std::string kernel_name("gemm_buffer_");
    if(TransA == CblasNoTrans && TransB == CblasNoTrans) {
        kernel_name += "NN";
        if(halfPrecisionMode) {
          sub_group_size = 16;
        }
    } else if(TransA == CblasNoTrans && TransB != CblasNoTrans) {
        if (M == 2)
          kernel_name +="NT_M_2";
        else if (M == 4)
          kernel_name +="NT_M_4";
        else if (M == 8)
          kernel_name +="NT_M_8";
        else
          kernel_name += "NT";
    } else if(TransA != CblasNoTrans && TransB == CblasNoTrans) {
        kernel_name += "TN";
        if(halfPrecisionMode) {
          sub_group_size = 16;
        }
    } else {
        kernel_name += "TT";
    }

    if(halfPrecisionMode) {
      kernel_name += "_half";
    } else {
      kernel_name += "_float";
    }

    oclk_gemm_float = &program.get_kernel(kernel_name);
    size_t local[2] = {};
    size_t global[2] = {};
    if (TransA == CblasNoTrans && TransB != CblasNoTrans && is_small_batch ) {
      if(M == 8)
        local[0] = 16;
      else if(M == 4)
        local[0] = 32;
      else
        local[0] = 64;
      local[1] = 1;

      if(M == 8)
        global[0] = N * local[0];
      else
        global[0] = (N + 3) / 4 * local[0];
      global[1] = 1;
    } else {
      size_t lx = sub_group_size;
      size_t ly = (TransB != CblasNoTrans && TransA == CblasNoTrans && halfPrecisionMode) ? 2 : 4;
      int dx = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 1 : 4;
      int dy = 8;
      size_t gx = (size_t)(N + dx - 1) / dx;
      size_t gy = (size_t)(M + dy - 1) / dy;
      global[0] = (gx + lx - 1) / lx * lx;
      global[1] = (gy + ly - 1) / ly * ly;
      local[0] = lx;
      local[1] = ly;
    }

    cl_uint arg_idx = 0;
    oclk_gemm_float->arg(arg_idx++, WrapHandle(A, &ctx));
    oclk_gemm_float->arg(arg_idx++, offA);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(B, &ctx));
    oclk_gemm_float->arg(arg_idx++, offB);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(C, &ctx));
    oclk_gemm_float->arg(arg_idx++, offC);
    oclk_gemm_float->arg(arg_idx++, M);
    oclk_gemm_float->arg(arg_idx++, N);
    oclk_gemm_float->arg(arg_idx++, K);
    oclk_gemm_float->arg(arg_idx++, fixup_arg_type(alpha));
    oclk_gemm_float->arg(arg_idx++, fixup_arg_type(beta));

    if(TransB == CblasNoTrans || TransA != CblasNoTrans) {
        int stride = 256;
        for(int start_index = 0; start_index < K; start_index += stride) {
            oclk_gemm_float->arg(arg_idx, start_index);
            OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                            oclk_gemm_float->handle().get(), 2, NULL,
                                            global, local, 0,
                                            NULL, &ev));
        }
    } else {
        OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                         oclk_gemm_float->handle().get(), 2, NULL,
                                         global, local, 0,
                                         NULL, &ev));
    }
    clReleaseEvent(ev);
}

template<typename Dtype>
static void innerprod_common(const int_tp ctx_id, const CBLAS_TRANSPOSE TransB,
                             const int_tp M, const int_tp N, const int_tp K,
                             const cl_mem A, const cl_mem B, const cl_mem B_image,
                             cl_mem C, gemm_type_t gemm_type, const size_t max_image_size) {

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
        gemm_type == GEMM_TYPE_FAST_IMAGE_32_2) {
      greentea_gpu_fast_image_gemm<Dtype>(ctx_id, CblasNoTrans, TransB, M, N, K,
                                   (Dtype)1., A, 0, B, 0, (Dtype)0., C,
                                   0, false, false, gemm_type, max_image_size);
    } else if (gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE) {
      greentea_gpu_fast_image_gemm<Dtype>(ctx_id, CblasNoTrans, TransB, M, N, K,
                                   (Dtype)1., A, 0, B_image, 0, (Dtype)0., C,
                                   0, false, true, GEMM_TYPE_FAST_IMAGE_B_IMAGE, max_image_size);

    } else if (gemm_type == GEMM_TYPE_FAST_BUFFER) {
      greentea_gpu_fast_buffer_gemm<Dtype>(ctx_id, CblasNoTrans, TransB, M, N, K,
                                    1.f, A, 0, B, 0, 0.f, C,
                                    0, gemm_type);
    } else
      greentea_gpu_gemm<Dtype>(ctx_id, CblasNoTrans, TransB, M, N, K,
                               (Dtype)1., A, 0, B, 0, (Dtype)0., C, 0);
}



template<typename Dtype>
void InnerProductLayer<Dtype>::generate_key() {
  std::stringstream keyBuilder;
  keyBuilder << M_ << "_"
             << N_ << "_"
             << K_ << "_"
             << transpose_;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                (this->device_->id());
  std::string prefix = ctx.current_device().name() + ctx.current_device().vendor()
                       + ctx.current_device().driver_version()
                       + std::to_string(ctx.current_device().max_compute_units());
  key_ = viennacl::tools::sha1(prefix + keyBuilder.str());
  // short_key_ = keyBuilder.str();
}
#ifdef HAS_HALF_SUPPORT
template void InnerProductLayer<half>::generate_key();
#endif
template void InnerProductLayer<float>::generate_key();
template void InnerProductLayer<double>::generate_key();

template<typename Dtype>
bool InnerProductLayer<Dtype>::load_cache() {
  if (tuned_)
    return true;
  else {
    generate_key();
    // Find cached kernel configuration
    string outputFile;
    outputFile = cache_path_.str() + key_;
    std::ifstream cachedKernel(outputFile.c_str());
    if (cachedKernel) {
      int cache_config;
      cachedKernel >> cache_config;
      innerprod_type_ = (gemm_type_t)cache_config;
      tuned_ = true;
      return true;
    } else {
      return false;
    }
  }
}

#ifdef HAS_HALF_SUPPORT
template bool InnerProductLayer<half>::load_cache();
#endif
template bool InnerProductLayer<float>::load_cache();
template bool InnerProductLayer<double>::load_cache();

template<typename Dtype>
void InnerProductLayer<Dtype>::tune_innerprod_type(const int_tp ctx_id,
                      const CBLAS_TRANSPOSE TransB, const cl_mem A, const cl_mem B,
                      const cl_mem B_image, const size_t max_image_size) {
  if (std::is_same<Dtype, double>::value) {
    innerprod_type_ = GEMM_TYPE_DEFAULT;
    return;
  } else {
    //1. load cache
    if (load_cache()) {
      return;
    } else {
      //2. if not cached generate tuning
      uint element_size = 0;
      bool halfPrecisionMode = !std::is_same<Dtype, float>::value;
      if(halfPrecisionMode) {
        element_size = sizeof(uint16_t);
      } else {
        element_size = sizeof(float);
      }
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
      cl_int err;

      cl_mem C = clCreateBuffer(ctx.handle().get(), CL_MEM_ALLOC_HOST_PTR, M_ * N_ * element_size, NULL, &err);
      OCL_CHECK(err);

      std::vector<gemm_type_t> gemm_tests;

      gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_32_1);
      if (B_image != NULL)
        gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_B_IMAGE);
      gemm_tests.push_back(GEMM_TYPE_FAST_BUFFER);
      if(!halfPrecisionMode)
        gemm_tests.push_back(GEMM_TYPE_DEFAULT);

      // warm up.
      for( int i = 0; i < gemm_tests.size(); i++ ) {
        innerprod_common<Dtype>(ctx_id, TransB, M_, N_, K_,
                         A, B, B_image, C, gemm_tests[i], max_image_size);
      }
      float fastest_time = 1e10;
      int fastest_index = -1;
      clFinish(ctx.get_queue().handle().get());
      for( int i = 0; i < gemm_tests.size(); i++ ) {
        Timer timer;
        timer.initted();
        timer.Start();
        innerprod_common<Dtype>(ctx_id, TransB, M_, N_, K_,
                         A, B, B_image, C, gemm_tests[i], max_image_size);
        timer.Stop();
        float elapsedTime = timer.MilliSeconds();
// #define INNERPROD_PROFILING
#ifdef INNERPROD_PROFILING
        std::cout << "innerprod type: " << gemm_tests[i] <<" eclipsed time: "
                  << elapsedTime << "ms." << std::endl;
#endif
        if (elapsedTime < fastest_time) {
          fastest_time = elapsedTime;
          fastest_index = i;
        }
      }
      clReleaseMemObject(C);

      if (fastest_index >= 0) {
        innerprod_type_ = gemm_tests[fastest_index];
      }
      //3. store cache.
      string outputFile;
      outputFile = cache_path_.str() + key_;
      std::ofstream outputKernel;
      outputKernel.open(outputFile.c_str());
      outputKernel << innerprod_type_;
      outputKernel.close();
      tuned_ = true;
      return;
    }
  }
  return;
}

#ifdef HAS_HALF_SUPPORT
template void InnerProductLayer<half>::tune_innerprod_type(const int_tp ctx_id,
                              const CBLAS_TRANSPOSE TransB, const cl_mem A, const cl_mem B,
                              const cl_mem B_image, const size_t max_image_size);
#endif
template void InnerProductLayer<float>::tune_innerprod_type(const int_tp ctx_id,
                              const CBLAS_TRANSPOSE TransB, const cl_mem A, const cl_mem B,
                              const cl_mem B_image, const size_t max_image_size);
template void InnerProductLayer<double>::tune_innerprod_type(const int_tp ctx_id,
                              const CBLAS_TRANSPOSE TransB, const cl_mem A, const cl_mem B,
                              const cl_mem B_image, const size_t max_image_size);

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (M_ == 1) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype) 1., weight,
                            bottom_data, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                              this->blobs_[1]->gpu_data(), top_data);
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, (Dtype) 1.,
                            bottom_data, weight, (Dtype) 0., top_data);
      if (bias_term_)
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                              bias_multiplier_.gpu_data(),
                              this->blobs_[1]->gpu_data(), (Dtype) 1.,
                              top_data);
    }
#endif  // USE CUDA
  } else {
#ifdef USE_GREENTEA
    int padded_height = 0, padded_width = 0;
    int height = !transpose_ ? N_ : K_;
    int width = !transpose_ ? K_ : N_;
    if (M_ != 1) {
      if (std::is_same<Dtype, float>::value) {
        padded_height = !transpose_ ? height : (height + ((height & 7) ? 1 : 0));
        padded_width = !transpose_ ? width : (width + ((width & 7) ? 1 : 0));
      } else {
        padded_height = !transpose_ ? height : (height + ((height & 7) ? (8-(height%8)) : 0));
        padded_width = !transpose_ ? width : (width + ((width & 7) ? (8-(width%8)) : 0));
      }
    }

    if (M_ == 1) {
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, N_,
                               K_, (Dtype) 1., (cl_mem) weight, 0,
                               (cl_mem) bottom_data, 0, (Dtype) 0.,
                               (cl_mem) top_data, 0);
      if (bias_term_)
        greentea_gpu_axpy<Dtype>(this->device_->id(), N_,
                                 bias_multiplier_.cpu_data()[0],
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (cl_mem) top_data, 0);
    } else {
      viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(this->device_->id());
      size_t max_image_size = std::min(ctx.devices()[0].image2d_max_width(),
                                       ctx.devices()[0].image2d_max_height());
      if (M_ <= max_image_size &&
          N_ <= max_image_size &&
          K_ <= max_image_size &&
          !std::is_same<Dtype, double>::value &&
          this->device_->CheckCapability("cl_intel_subgroups")) {
        if (!test_only_ || copied_weight_data_ != this->blobs_[0]->data().get()) {
          int height = !transpose_ ? N_ : K_;
          int width = !transpose_ ? K_ : N_;
          if (weight_image_) {
            clReleaseMemObject((cl_mem)weight_image_);
            weight_image_ = NULL;
          }
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(this->device_->id(),
            &weight_image_, (cl_mem) weight, 0,
            false, !transpose_,
            true, padded_height, padded_width,
            height, width, width, (int)0, NULL, NULL);
          copied_weight_data_ = this->blobs_[0]->data().get();
        }
      }

      tune_innerprod_type(this->device_->id(),
                          transpose_ ? CblasNoTrans : CblasTrans,
                          (cl_mem) bottom_data, (cl_mem) weight, (cl_mem) weight_image_,
                          max_image_size);

      innerprod_common<Dtype>(this->device_->id(),
                       transpose_ ? CblasNoTrans : CblasTrans,
                       M_, N_, K_, (cl_mem) bottom_data,
                       (cl_mem) weight, (cl_mem) weight_image_,
                       (cl_mem) top_data, innerprod_type_, max_image_size);
      if (bias_term_) {
        // Execute kernel
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                 (cl_mem) (bias_multiplier_.gpu_data()), 0,
                                 (cl_mem) (this->blobs_[1]->gpu_data()), 0,
                                 (Dtype) 1., (cl_mem) top_data, 0);
      }
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  test_only_ = false;
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                              (Dtype) 1.,
                              bottom_data, top_diff, (Dtype) 1.,
                              this->blobs_[0]->mutable_gpu_diff());
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                              (Dtype) 1.,
                              top_diff, bottom_data, (Dtype) 1.,
                              this->blobs_[0]->mutable_gpu_diff());
      }
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
                            bias_multiplier_.gpu_data(), (Dtype) 1.,
                            this->blobs_[1]->mutable_gpu_diff());
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Gradient with respect to weight
      if (transpose_) {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans, CblasNoTrans,
                                 K_, N_, M_, (Dtype) 1., (cl_mem) bottom_data,
                                 0, (cl_mem) top_diff, 0, (Dtype) 1.,
                                 (cl_mem) (this->blobs_[0]->mutable_gpu_diff()),
                                 0);
      } else {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans, CblasNoTrans,
                                 N_, K_, M_, (Dtype) 1., (cl_mem) top_diff, 0,
                                 (cl_mem) bottom_data, 0, (Dtype) 1.,
                                 (cl_mem) (this->blobs_[0]->mutable_gpu_diff()),
                                 0);
      }
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans, M_, N_,
                               (Dtype) 1., (cl_mem) top_diff, 0,
                               (cl_mem) (bias_multiplier_.gpu_data()), 0,
                               (Dtype) 1.,
                               (cl_mem) (this->blobs_[1]->mutable_gpu_diff()),
                               0);
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      if (transpose_) {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasTrans, M_, K_, N_, (Dtype) 1.,
                                 (cl_mem) top_diff, 0,
                                 (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                                 (Dtype) 0.,
                                 (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
      } else {
        greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                                 CblasNoTrans, M_, K_, N_, (Dtype) 1.,
                                 (cl_mem) top_diff, 0,
                                 (cl_mem) (this->blobs_[0]->gpu_data()), 0,
                                 (Dtype) 0.,
                                 (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
      }
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
