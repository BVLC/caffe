#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef TODO_REFACTOR

#ifdef USE_OPENCL
  cl_mem weight_image_;
  const SyncedMemory * copied_weight_data_;
  bool test_only_;
  uint64_t weight_image_seq_;
  gemm_type_t innerprod_type_;
  bool tuned_;
  stringstream cache_path_;
  string key_;
#endif

#ifdef USE_OPENCL
  virtual void generate_key();
  virtual void tune_innerprod_type(const int_tp ctx_id,
       const CBLAS_TRANSPOSE trans_b, const cl_mem a,
       const cl_mem b, const cl_mem B_image, const size_t max_image_size);
  virtual bool load_cache();
#endif

#ifdef USE_OPENCL
  ~InnerProductLayer() {
    if (weight_image_)
      clReleaseMemObject(weight_image_);
    weight_image_ = NULL;
  }
#endif

enum gemm_type_t {
  GEMM_TYPE_DEFAULT = 0,
  GEMM_TYPE_FAST_IMAGE_32_1,
  GEMM_TYPE_FAST_IMAGE_32_2,
  GEMM_TYPE_FAST_IMAGE_B_IMAGE,
  GEMM_TYPE_FAST_BUFFER
};

struct gemm_callback_arg {
  vector<cl_event> evs;
  vector<cl_mem> imgs;
};

static void CL_CALLBACK gemm_callback(cl_event event,
                                cl_int event_command_exec_status,
                                void *user_data) {
  struct gemm_callback_arg *arg = (struct gemm_callback_arg *) user_data;
  for (int i = 0; i < arg->evs.size(); i++) {
    clReleaseEvent(arg->evs[i]);
  }

  for (int i = 0; i < arg->imgs.size(); i++) {
    clReleaseMemObject(arg->imgs[i]);
  }
  delete arg;
}

// Create and copy buffer to image for GEMM's matrix a and b.
// Will return image to caller if the input image is NULL. Otherwise,
// will use the image directly. It's caller's responsibility to
// release the created image.
template<typename Dtype, typename MItype, typename MOtype>
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
  int src_offset = sizeof(Dtype) * offset;
  if (!is_matrix_a && transpose) {
  // For matrix b with transpose, we need to handle them differently.
  // As we can't use the sub group block read to get a row easily,
  // we have to use CL_FLOAT type with read_imagef to get the row.
    cl_int err;
    if (halfPrecisionMode) {
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

    if (ld == width) {
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
      if (halfPrecisionMode) {
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
                                 origin, region, wait_list_size,
                                 wait_list, event));
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

template<typename Dtype, typename MItype, typename MOtype>
static void greentea_gpu_fast_image_gemm(const int_tp ctx_id,
                       const CBLAS_TRANSPOSE trans_a,
                       const CBLAS_TRANSPOSE trans_b, const int_tp m,
                       const int_tp n, const int_tp k, const Dtype alpha,
                       const cl_mem a, const int_tp offA, const cl_mem b,
                       const int_tp offB, const Dtype beta, cl_mem c,
                       const int_tp offC, bool is_image_a, bool is_image_b,
                       enum gemm_type_t gemm_type,
                       const size_t max_image_size) {
  CHECK_EQ(gemm_type == GEMM_TYPE_FAST_IMAGE_32_1
           || gemm_type == GEMM_TYPE_FAST_IMAGE_32_2
           || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE, true)
    << "Invalid fast image gemm type." << std::endl;
  if (is_image_a)
    CHECK_EQ(offA, 0) << "Invalid input image offset." << std::endl;

  if (is_image_b)
    CHECK_EQ(offB, 0) << "Invalid input image offset." << std::endl;

  bool halfPrecisionMode = !std::is_same<Dtype, float>::value;
  int widthA = (trans_a == CblasNoTrans) ? k : m;
  int heightA = (trans_a == CblasNoTrans) ? m : k;
  int widthB = (trans_b == CblasNoTrans) ? n : k;
  int heightB = (trans_b == CblasNoTrans) ? k : n;

  int ldA = widthA;
  int ldB = widthB;
  int ldC = n;

  int A_start_x = 0, A_start_y = 0, B_start_x = 0;
  int B_start_y = 0, C_start_x = 0, C_start_y = 0;
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
  string kernel_name("gemm_");
  if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1
      || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
    kernel_name += "32_1_";
  else
    kernel_name += "32_2_";

  if (trans_a == CblasNoTrans)
    kernel_name += "n";
  else
    kernel_name += "T";

  if (trans_b == CblasNoTrans) {
    kernel_name += "N_";
  } else {
    kernel_name += "T_";
    if (is_image_b || (k % use_buffer_indicator != 0)) {
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

  if (halfPrecisionMode) {
    kernel_name += "_half";
  } else {
    kernel_name += "_float";
  }

  oclk_gemm_float = &program.get_kernel(kernel_name);
  while (C_start_y < m) {
    blockC_width = std::min(static_cast<int>(n) - C_start_x, blocksize);
    blockC_height = std::min(static_cast<int>(m) - C_start_y, blocksize);

    int isFirstColBlock = 1;
    for (int k = 0; k < k; k += blocksize) {
      cl_event ev[5];
      cl_uint ev_idx = 0;
      memset(ev, 0, sizeof(cl_event) * 5);
      struct gemm_callback_arg * arg = new gemm_callback_arg;

      blockA_width = std::min(widthA - A_start_x, blocksize);
      blockA_height = std::min(heightA - A_start_y, blocksize);
      blockB_width = std::min(widthB - B_start_x, blocksize);
      blockB_height = std::min(heightB - B_start_y, blocksize);
      int block_Ksize = std::min(static_cast<int>(k) - k, blocksize);

      int padded_k = block_Ksize + ((block_Ksize & 7) ?
                       (8 - (block_Ksize & 7)) : 0);
      int imageA_w = (trans_a == CblasNoTrans) ? padded_k : blockA_width;
      int imageA_h = (trans_a == CblasNoTrans) ? blockA_height : padded_k;
      int imageB_w = (trans_b == CblasNoTrans) ? blockB_width : padded_k;
      int imageB_h = (trans_b == CblasNoTrans) ? padded_k : blockB_height;

      int blockA_offset = offA + A_start_y * ldA + A_start_x;
      int blockB_offset = offB + B_start_y * ldB + B_start_x;
      int blockC_offset = offC + C_start_y * ldC + C_start_x;
      if (trans_b == CblasNoTrans) {
        bool padding_A = false;
        bool padding_B = false;

        if (halfPrecisionMode && is_image_b) {
          padding_A = true;
        }

        if (!is_image_a && !is_image_b) {
          if (m * k < n * k)
            padding_B = true;
          else
            padding_A = true;
        }

        if (!is_image_a) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImA,
                                    a, blockA_offset,
                                    true, trans_a != CblasNoTrans,
                                    padding_A, imageA_h, imageA_w,
                                    blockA_height, blockA_width, ldA, 0,
                                    NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
        if (!is_image_b) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImB,
                                    b, blockB_offset,
                                    false, false,
                                    padding_B, imageB_h, imageB_w,
                                    blockB_height, blockB_width, ldB,
                                    0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
      } else {
        // We will use normal read_imagef to read image b when b has transpose.
        // thus we don't need to pad image a at all.
        if (!is_image_a) {
          bool padding;
          padding = !is_image_b || halfPrecisionMode;
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImA,
                                    a, blockA_offset,
                                    true, trans_a != CblasNoTrans,
                                    padding, imageA_h, imageA_w,
                                    blockA_height, blockA_width, ldA,
                                    0, NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }

        if (!is_image_b && (k % use_buffer_indicator != 0)) {
          greentea_gpu_gemm_copy_buffer_to_image<Dtype>(ctx_id, &ImB,
                                    b, blockB_offset,
                                    false, true, false, imageB_h, imageB_w,
                                    blockB_height, blockB_width, ldB, 0,
                                    NULL, &ev[ev_idx]);
          if (ev[ev_idx] != NULL)
            ev_idx++;
        }
      }
      if (is_image_a)
        ImA = a;
      if (is_image_b)
        ImB = b;

      size_t global[2];
      if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
          gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE ) {
        if (halfPrecisionMode) {
          global[0] = (size_t)( blockC_width + 15 ) & ~15;
        } else {
          global[0] = (size_t)( blockC_width + 7 ) & ~7;
        }
      } else {
        if (halfPrecisionMode) {
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
      if (trans_b == CblasNoTrans || is_image_b ||
          (k % use_buffer_indicator != 0)) {
        oclk_gemm_float->arg(arg_idx++, WrapHandle(ImB, &ctx));
      } else {
        oclk_gemm_float->arg(arg_idx++, WrapHandle(b, &ctx));
        oclk_gemm_float->arg(arg_idx++, blockB_offset);
        oclk_gemm_float->arg(arg_idx++, ldB);
      }
      oclk_gemm_float->arg(arg_idx++, WrapHandle(c, &ctx));
      oclk_gemm_float->arg(arg_idx++, blockC_offset);
      oclk_gemm_float->arg(arg_idx++, blockC_height);
      oclk_gemm_float->arg(arg_idx++, blockC_width);
      oclk_gemm_float->arg(arg_idx++, ldC);
      oclk_gemm_float->arg(arg_idx++, fixup_arg_type(alpha));
      oclk_gemm_float->arg(arg_idx++, fixup_arg_type(beta));
      oclk_gemm_float->arg(arg_idx++, padded_k);
      if (trans_b != CblasNoTrans)
        oclk_gemm_float->arg(arg_idx++, block_Ksize);
      oclk_gemm_float->arg(arg_idx++, isFirstColBlock);

      cl_event *wait_list = NULL;
      if (ev_idx != 0)
        wait_list = &ev[0];
      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_gemm_float->handle().get(), 2, NULL,
                                       global, local, ev_idx,
                                       wait_list, &ev[ev_idx]));
      if (trans_a == CblasNoTrans)
        A_start_x += blockA_width;
      else
        A_start_y += blockA_height;

      if (trans_b == CblasNoTrans)
        B_start_y += blockB_height;
      else
        B_start_x += blockB_width;

      isFirstColBlock = 0;
      arg->evs.assign(ev, ev + ev_idx + 1);
      clSetEventCallback(ev[ev_idx], CL_COMPLETE, &gemm_callback,
                         static_cast<void*>(arg));
    }

    C_start_x += blockC_width;
    if (trans_a == CblasNoTrans)
      A_start_x = 0;
    else
      A_start_y = 0;
    if (trans_b == CblasNoTrans) {
      B_start_x += blockB_width;
      B_start_y = 0;
    } else {
      B_start_y += blockB_height;
      B_start_x = 0;
    }
    if (C_start_x >= n) {
      C_start_x = 0;
      B_start_x = 0;
      B_start_y = 0;
      C_start_y += blockC_height;
      if (trans_a == CblasNoTrans)
        A_start_y += blockA_height;
      else
        A_start_x += blockA_width;
    }
  }

  if (ImA && !is_image_a)
    clReleaseMemObject(ImA);
  if (ImB && !is_image_b)
    clReleaseMemObject(ImB);
}

template<typename Dtype, typename MItype, typename MOtype>
static void greentea_gpu_fast_buffer_gemm(const int_tp ctx_id,
                       const CBLAS_TRANSPOSE trans_a,
                       const CBLAS_TRANSPOSE trans_b, const int_tp m,
                       const int_tp n, const int_tp k, const Dtype alpha,
                       const cl_mem a, const int_tp offA, const cl_mem b,
                       const int_tp offB, const Dtype beta, cl_mem c,
                       const int_tp offC, enum gemm_type_t gemm_type) {
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_BUFFER, true)
      << "Invalid fast buffer gemm type." << std::endl;

    cl_event ev;

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
    viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
                                       ->program();
    bool halfPrecisionMode = !std::is_same<Dtype, float>::value;

    size_t sub_group_size = 8;
    bool is_small_batch = (m == 2 || m == 4 || m == 8);
    viennacl::ocl::kernel *oclk_gemm_float;
    string kernel_name("gemm_buffer_");
    if (trans_a == CblasNoTrans && trans_b == CblasNoTrans) {
        kernel_name += "NN";
        if (halfPrecisionMode) {
          sub_group_size = 16;
        }
    } else if (trans_a == CblasNoTrans && trans_b != CblasNoTrans) {
        if (m == 2)
          kernel_name +="NT_M_2";
        else if (m == 4)
          kernel_name +="NT_M_4";
        else if (m == 8)
          kernel_name +="NT_M_8";
        else
          kernel_name += "NT";
    } else if (trans_a != CblasNoTrans && trans_b == CblasNoTrans) {
        kernel_name += "TN";
        if (halfPrecisionMode) {
          sub_group_size = 16;
        }
    } else {
        kernel_name += "TT";
    }

    if (halfPrecisionMode) {
      kernel_name += "_half";
    } else {
      kernel_name += "_float";
    }

    oclk_gemm_float = &program.get_kernel(kernel_name);
    size_t local[2] = {};
    size_t global[2] = {};
    if (trans_a == CblasNoTrans && trans_b != CblasNoTrans && is_small_batch) {
      if (m == 8)
        local[0] = 16;
      else if (m == 4)
        local[0] = 32;
      else
        local[0] = 64;
      local[1] = 1;

      if (m == 8)
        global[0] = n * local[0];
      else
        global[0] = (n + 3) / 4 * local[0];
      global[1] = 1;
    } else {
      size_t lx = sub_group_size;
      size_t ly = (trans_b != CblasNoTrans &&
                  trans_a == CblasNoTrans && halfPrecisionMode) ? 2 : 4;
      int dx = (trans_b != CblasNoTrans && trans_a == CblasNoTrans) ? 1 : 4;
      int dy = 8;
      size_t gx = (size_t)(n + dx - 1) / dx;
      size_t gy = (size_t)(m + dy - 1) / dy;
      global[0] = (gx + lx - 1) / lx * lx;
      global[1] = (gy + ly - 1) / ly * ly;
      local[0] = lx;
      local[1] = ly;
    }

    cl_uint arg_idx = 0;
    oclk_gemm_float->arg(arg_idx++, WrapHandle(a, &ctx));
    oclk_gemm_float->arg(arg_idx++, offA);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(b, &ctx));
    oclk_gemm_float->arg(arg_idx++, offB);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(c, &ctx));
    oclk_gemm_float->arg(arg_idx++, offC);
    oclk_gemm_float->arg(arg_idx++, m);
    oclk_gemm_float->arg(arg_idx++, n);
    oclk_gemm_float->arg(arg_idx++, k);
    oclk_gemm_float->arg(arg_idx++, fixup_arg_type(alpha));
    oclk_gemm_float->arg(arg_idx++, fixup_arg_type(beta));

    if (trans_b == CblasNoTrans || trans_a != CblasNoTrans) {
        int stride = 256;
        for (int start_index = 0; start_index < k; start_index += stride) {
            oclk_gemm_float->arg(arg_idx, start_index);
            OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                             oclk_gemm_float->handle().get(),
                                             2, NULL,
                                             global, local, 0,
                                             NULL, &ev));
        }
    } else {
        OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                         oclk_gemm_float->handle().get(),
                                         2, NULL,
                                         global, local, 0,
                                         NULL, &ev));
    }
    clReleaseEvent(ev);
}

template<typename Dtype, typename MItype, typename MOtype>
static void innerprod_common(const int_tp ctx_id, const CBLAS_TRANSPOSE trans_b,
                             const int_tp m, const int_tp n, const int_tp k,
                             const cl_mem a, const cl_mem b,
                             const cl_mem B_image,
                             cl_mem c, gemm_type_t gemm_type,
                             const size_t max_image_size) {
    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
        gemm_type == GEMM_TYPE_FAST_IMAGE_32_2) {
      greentea_gpu_fast_image_gemm<Dtype>(ctx_id, CblasNoTrans, trans_b, m, n, k,
                                   (Dtype)1., a, 0, b, 0, (Dtype)0., c,
                                   0, false, false, gemm_type, max_image_size);
    } else if (gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE) {
      greentea_gpu_fast_image_gemm<Dtype>(ctx_id, CblasNoTrans, trans_b, m, n, k,
                                   (Dtype)1., a, 0, B_image, 0, (Dtype)0., c,
                                   0, false, true,
                                   GEMM_TYPE_FAST_IMAGE_B_IMAGE,
                                   max_image_size);

    } else if (gemm_type == GEMM_TYPE_FAST_BUFFER) {
      greentea_gpu_fast_buffer_gemm<Dtype>(ctx_id, CblasNoTrans,
                                    trans_b, m, n, k,
                                    1.f, a, 0, b, 0, 0.f, c,
                                    0, gemm_type);
    } else {
      greentea_gpu_gemm<Dtype>(ctx_id, CblasNoTrans, trans_b, m, n, k,
                               (Dtype)1., a, 0, b, 0, (Dtype)0., c, 0);
    }
}



template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::generate_key() {
  stringstream keyBuilder;
  keyBuilder << M_ << "_"
             << N_ << "_"
             << K_ << "_"
             << transpose_;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                (this->device_->id());
  string prefix = ctx.current_device().name()
                   + ctx.current_device().vendor()
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

template<typename Dtype, typename MItype, typename MOtype>
bool InnerProductLayer<Dtype, MItype, MOtype>::load_cache() {
  if (tuned_) {
    return true;
  } else {
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

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::tune_innerprod_type(const int_tp ctx_id,
                      const CBLAS_TRANSPOSE trans_b, const cl_mem a,
                      const cl_mem b,
                      const cl_mem B_image, const size_t max_image_size) {
  if (std::is_same<Dtype, double>::value) {
    innerprod_type_ = GEMM_TYPE_DEFAULT;
    return;
  } else {
    // 1. load cache
    if (load_cache()) {
      return;
    } else {
      // 2. if not cached generate tuning
      uint element_size = 0;
      bool halfPrecisionMode = !std::is_same<Dtype, float>::value;
      if (halfPrecisionMode) {
        element_size = sizeof(uint16_t);
      } else {
        element_size = sizeof(float);
      }
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
      cl_int err;

      cl_mem c = clCreateBuffer(ctx.handle().get(),
                                CL_MEM_ALLOC_HOST_PTR,
                                M_ * N_ * element_size, NULL, &err);
      OCL_CHECK(err);

      vector<gemm_type_t> gemm_tests;

      gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_32_1);
      if (B_image != NULL)
        gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_B_IMAGE);
      gemm_tests.push_back(GEMM_TYPE_FAST_BUFFER);
      if (!halfPrecisionMode)
        gemm_tests.push_back(GEMM_TYPE_DEFAULT);

      // warm up.
      for ( int i = 0; i < gemm_tests.size(); i++ ) {
        innerprod_common<Dtype>(ctx_id, trans_b, M_, N_, K_,
                         a, b, B_image, c, gemm_tests[i], max_image_size);
      }
      float fastest_time = 1e10;
      int fastest_index = -1;
      clFinish(ctx.get_queue().handle().get());
      for ( int i = 0; i < gemm_tests.size(); i++ ) {
        Timer timer;
        timer.initted();
        timer.Start();
        innerprod_common<Dtype>(ctx_id, trans_b, M_, N_, K_,
                         a, b, B_image, c, gemm_tests[i], max_image_size);
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
      clReleaseMemObject(c);

      if (fastest_index >= 0) {
        innerprod_type_ = gemm_tests[fastest_index];
      }
      // 3. store cache.
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
                              const CBLAS_TRANSPOSE trans_b, const cl_mem a,
                              const cl_mem b, const cl_mem B_image,
                              const size_t max_image_size);
#endif
template void InnerProductLayer<float>::tune_innerprod_type(const int_tp ctx_id,
                              const CBLAS_TRANSPOSE trans_b, const cl_mem a,
                              const cl_mem b, const cl_mem B_image,
                              const size_t max_image_size);
template void InnerProductLayer<double>::tune_innerprod_type(
                              const int_tp ctx_id,
                              const CBLAS_TRANSPOSE trans_b, const cl_mem a,
                              const cl_mem b, const cl_mem B_image,
                              const size_t max_image_size);


template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                           const vector<Blob<MItype>*>& bottom,
                                           const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();

  if (M_ == 1) {
    this->device_->template gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype) 1.,
                                     weight, bottom_data, (Dtype) 0., top_data);
    if (bias_term_)
      this->device_->template axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                                 this->blobs_[1]->gpu_data(), top_data);
  } else {
    this->device_->template gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype) 1.,
                          bottom_data, weight, (Dtype) 0., top_data);
    if (bias_term_)
      this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                            (Dtype) 1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype) 1.,
                            top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void InnerProductLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      this->device_->template gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                                          (Dtype) 1.,
                                          bottom_data, top_diff, (Dtype) 1.,
                                          this->blobs_[0]->mutable_gpu_diff());
    } else {
      this->device_->template gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                                          (Dtype) 1.,
                                          top_diff, bottom_data, (Dtype) 1.,
                                          this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    this->device_->template gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1.,
                              top_diff, bias_multiplier_.gpu_data(), (Dtype) 1.,
                              this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      this->device_->template gemm<Dtype>(CblasNoTrans, CblasTrans,
                            M_, K_, N_,
                            (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                            (Dtype) 0., bottom[0]->mutable_gpu_diff());
    } else {
      this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            M_, K_, N_,
                            (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                            (Dtype) 0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(InnerProductLayer);

#endif  // TODO_REFACTOR

}  // namespace caffe
