#include <algorithm>
#include <cfloat>

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {
using std::min;
using std::max;

template <typename Dtype>
PoolingCodeGeneratorForward<Dtype>::PoolingCodeGeneratorForward() {
  Callback = NULL;
}

template <typename Dtype>
PoolingCodeGeneratorForward<Dtype>::~PoolingCodeGeneratorForward() {}

template <typename Dtype>
typename PoolingCodeGeneratorForward<Dtype>::Callback_t
    PoolingCodeGeneratorForward<Dtype>::Get_callback(
  PoolingLayer<Dtype>* layer,
  Blob<Dtype>* top,
  bool use_top_mask) {
  // Wrapper for lazy initialization.
  // Also check if top shape din't change.
  // TODO: do we need to check all blobs' shapes?
  // In future we may add cache for all already found options.
  // Currently there is only one code for last used shape.
  if (Callback == NULL ||
      top->shape() != Layer_output_shape_signature ||
      Use_top_mask != use_top_mask ||
      Method != layer->layer_param_.pooling_param().pool()) {
    Method = layer->layer_param_.pooling_param().pool();
    Use_top_mask = use_top_mask;
    Layer_output_shape_signature = top->shape();
    Create_callback(layer);
  }
  return Callback;
}

// Implementation of CodeGenerator classes for Pooling.
template <typename Dtype>
void PoolingCodeGeneratorForward<Dtype>::Naive(
  const Dtype* bottom_data,
  Dtype* top_data,
  int top_count,
  int batch_start,
  int batch_end,
  void* mask_ptr,
  PoolingLayer<Dtype>* layer,
  bool use_top_mask) {
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = static_cast<Dtype*>(mask_ptr);

  int pooled_fm_size = layer->pooled_height_ * layer->pooled_width_;
  int fm_size = layer->height_ * layer->width_;

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (layer->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (!use_top_mask) {
      mask = static_cast<int*>(mask_ptr);
    }

    bottom_data += fm_size*layer->channels_*batch_start;
    top_data += pooled_fm_size*layer->channels_*batch_start;
    if (use_top_mask) {
      top_mask += pooled_fm_size*layer->channels_*batch_start;
    } else {
      mask += pooled_fm_size*layer->channels_*batch_start;
    }

    // The main loop
    for (int n = batch_start; n < batch_end; ++n) {
      for (int c = 0; c < layer->channels_; ++c) {
        for (int ph = 0; ph < layer->pooled_height_; ++ph) {
          for (int pw = 0; pw < layer->pooled_width_; ++pw) {
            int hstart = ph * layer->stride_h_ - layer->pad_h_;
            int wstart = pw * layer->stride_w_ - layer->pad_w_;
            int hend = min(hstart + layer->kernel_h_, layer->height_);
            int wend = min(wstart + layer->kernel_w_, layer->width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * layer->pooled_width_ + pw;
            Dtype acc = -FLT_MAX;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * layer->width_ + w;
                if (bottom_data[index] > acc) {
                  acc = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
            top_data[pool_index] = acc;
          }
        }
        // compute offset
        bottom_data += fm_size;
        top_data += pooled_fm_size;
        if (use_top_mask) {
          top_mask += pooled_fm_size;
        } else {
          mask += pooled_fm_size;
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:

    bottom_data += fm_size*layer->channels_*batch_start;
    top_data += pooled_fm_size*layer->channels_*batch_start;

    // The main loop
    for (int n = batch_start; n < batch_end; ++n) {
      for (int c = 0; c < layer->channels_; ++c) {
        for (int ph = 0; ph < layer->pooled_height_; ++ph) {
          for (int pw = 0; pw < layer->pooled_width_; ++pw) {
            int hstart = ph * layer->stride_h_ - layer->pad_h_;
            int wstart = pw * layer->stride_w_ - layer->pad_w_;
            int hend =
              min(hstart + layer->kernel_h_, layer->height_ + layer->pad_h_);
            int wend =
              min(wstart + layer->kernel_w_, layer->width_ + layer->pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, layer->height_);
            wend = min(wend, layer->width_);
            Dtype acc = 0;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                acc += bottom_data[h * layer->width_ + w];
              }
            }
            top_data[ph * layer->pooled_width_ + pw] = acc / pool_size;
          }
        }
        // compute offset
        bottom_data += fm_size;
        top_data += pooled_fm_size;
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

// Generic datatypes - use naive versions.
template <typename Dtype>
void PoolingCodeGeneratorForward<Dtype>::Create_callback(
  PoolingLayer<Dtype>* layer) {
  Callback = &Naive;
}

#if defined __x86_64__ || defined _M_X64
// Here we have specialized versions for supported formats in x64 architectures.
template <>
void PoolingCodeGeneratorForward<float>::Create_callback(
  PoolingLayer<float>* layer) {
  using Xbyak::util::Cpu;
  using Xbyak::Reg64;
  using Xbyak::Reg32;
  using Xbyak::Address;
  Cpu Current_cpu;
  const LayerParameter& param = layer->layer_param();
  if (Current_cpu.has(Cpu::tAVX2) &&
      (param.pooling_param().pool() == PoolingParameter_PoolMethod_AVE ||
       param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)) {
    // AVX2 optimized version.
    // Runtime constants.
    const int pooled_fm_size = layer->pooled_height_ * layer->pooled_width_;
    const int fm_size = layer->height_ * layer->width_;

    const int pooled_batch_size = pooled_fm_size * layer->channels_;
    const int batch_size = fm_size * layer->channels_;

    const int height_kernel_h_ = layer->height_ - layer->kernel_h_;
    const int width_kernel_w_ = layer->width_ - layer->kernel_w_;

    const int height_kernel_h_pad_h_ = height_kernel_h_ + layer->pad_h_;
    const int width_kernel_w_pad_w_ = width_kernel_w_ + layer->pad_w_;

    int64_t internal_mask_ptr =
      (Use_top_mask
        || param.pooling_param().pool() == PoolingParameter_PoolMethod_AVE) ?
          0 : reinterpret_cast<int64_t>(layer->max_idx_.cpu_data());

    bool optimal_version = false;

    if (layer->pad_h_ == 0 &&
        layer->pad_w_ == 0 &&
        (layer->pooled_height_-1) * layer->stride_h_ + layer->kernel_h_
         == layer->height_ &&
        (layer->pooled_width_-1) * layer->stride_w_ + layer->kernel_w_
         == layer->width_)
      optimal_version = true;

    // Register names.
    const Reg64& reg_input_ptr = rsi;
    const Reg64& reg_index_acc = rcx;
    const Reg32& reg_index_acc_l = ecx;
    const Reg64& reg_index_cnt = rdi;

    const Reg64& reg_scratch0 = r8;
    const Reg64& reg_scratch1 = r9;
    const Reg64& reg_scratch2 = r10;
    const Reg64& reg_scratch3 = r11;
    const Reg64& reg_scratch4 = r12;
    const Reg64& reg_scratch5 = r13;
    const Reg64& reg_scratch6 = r14;
    const Reg64& reg_scratch7 = r15;
    const Reg64& reg_scratch8 = rbx;

    const Reg64& reg_arg0 = rdi;
    const Reg64& reg_arg1 = rsi;
    const Reg64& reg_arg2 = rdx;
    const Reg64& reg_arg3 = rcx;
    const Reg64& reg_arg4 = r8;
    const Reg64& reg_arg5 = r9;

    const Reg64& reg_mul_param = rax;
    const Reg64& reg_mul_result_l = rax;

    // Stack variable names residing inside red zone.
    int stack_qwords = 0;
    const Address& stack_bottom_data_ptr = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_top_data_ptr    = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_top_count       = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_batch_start     = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_batch_end       = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_top_mask_ptr    = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_batch_cnt       = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_channel_cnt     = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_out_w_cnt       = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_out_h_cnt       = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_pool_index      = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_top_mask_orig   = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_top_orig        = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_bottom_orig     = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_min_float       = dword[rbp - (++stack_qwords * 8)];
    const Address& stack_wstart0         = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_index0          = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_true_height     = qword[rbp - (++stack_qwords * 8)];
    const Address& stack_true_width      = qword[rbp - (++stack_qwords * 8)];

    // ASSEMBLY STARTS HERE.
    // It seems we are regenerating the code due to output reshape
    if (Callback)
      reset();

    // Prologue.
    push(rbp);
    mov(rbp, rsp);
    sub(rsp, stack_qwords * 8);

    // Save r12-r15 and rbx registers.
    push(r12); push(r13); push(r14); push(r15); push(rbx);

    // Prepare default accumulator. (and bypass memory aliasing warning...)
    float min_float = -FLT_MAX;
    void* min_float_ptr = &min_float;
    uint32_t min_float_cast = *reinterpret_cast<uint32_t*>(min_float_ptr);
    mov(stack_min_float, min_float_cast);
    vbroadcastss(xmm15, stack_min_float);

    // Move register arguments to the stack,
    // we gonna need the registers for other purposes.
    mov(stack_bottom_orig, reg_arg0);
    mov(stack_top_orig,    reg_arg1);
    mov(stack_top_count,   reg_arg2);
    mov(stack_batch_start, reg_arg3);
    mov(stack_batch_end,   reg_arg4);

    if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
      if (Use_top_mask) {
        mov(stack_top_mask_orig, reg_arg5);
      } else {
        mov(reg_scratch0, internal_mask_ptr);
        mov(stack_top_mask_orig, reg_scratch0);
      }
    }

    // Further arguments on stack
    // (layer_ptr and use_top_mask) are ingored in this implementation.

    // Iterate through batches.
    mov(reg_scratch0, stack_batch_start);
    mov(stack_batch_cnt, reg_scratch0);
    L("batch_loop_start");
    mov(reg_scratch0, stack_batch_end);
    cmp(stack_batch_cnt, reg_scratch0);
    jae("batch_loop_end", T_NEAR);

      // Iterate through channels.
      mov(stack_channel_cnt, 0);
      L("channel_loop_start");
      cmp(stack_channel_cnt, layer->channels_);
      jae("channel_loop_end", T_NEAR);

        // Compute batch/channel offsets for buffers.
        // input: (batch_size*batch+fm_size*fm)*4 + ptr
        mov(reg_mul_param, batch_size);
        mul(stack_batch_cnt);
        mov(reg_scratch0, reg_mul_result_l);
        mov(reg_mul_param, fm_size);
        mul(stack_channel_cnt);
        add(reg_scratch0, reg_mul_result_l);
        shl(reg_scratch0, 2);
        add(reg_scratch0, stack_bottom_orig);
        mov(stack_bottom_data_ptr, reg_scratch0);

        // output: (pooled_batch_size*batch+pooled_fm_size*fm)*4 + ptr
        // mask same
        mov(reg_mul_param, pooled_batch_size);
        mul(stack_batch_cnt);
        mov(reg_scratch0, reg_mul_result_l);
        mov(reg_mul_param, pooled_fm_size);
        mul(stack_channel_cnt);
        add(reg_scratch0, reg_mul_result_l);
        shl(reg_scratch0, 2);
        mov(reg_scratch1, reg_scratch0);

        add(reg_scratch0, stack_top_orig);
        mov(stack_top_data_ptr, reg_scratch0);

        if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
          add(reg_scratch1, stack_top_mask_orig);
          mov(stack_top_mask_ptr, reg_scratch1);
        }

        // Iterate through output height.
        mov(stack_out_h_cnt, 0);
        L("out_h_loop_start");
        cmp(stack_out_h_cnt, layer->pooled_height_);
        jae("out_h_loop_end", T_NEAR);

          if (optimal_version) {
            // index_in = ph * stride_h_ * width_; stored in reg_scratch7
            mov(reg_mul_param, layer->stride_h_ * layer->width_);
            imul(stack_out_h_cnt);
            mov(reg_scratch7, reg_mul_result_l);

            // effective_in = index_in*4 + input_ptr; stored in reg_scratch5
            mov(reg_scratch0, stack_bottom_data_ptr);
            lea(reg_scratch5, ptr[reg_scratch7*4 + reg_scratch0]);

            // index_out = ph * pooled_width_; stored in reg_scratch6
            mov(reg_mul_param, layer->pooled_width_);
            imul(stack_out_h_cnt);
            mov(reg_scratch6, reg_mul_result_l);

            // effective_out = index_out*4 + output_ptr; stored in reg_scratch4
            mov(reg_scratch0, stack_top_data_ptr);
            lea(reg_scratch4, ptr[reg_scratch6*4 + reg_scratch0]);

            if (param.pooling_param().pool()
                == PoolingParameter_PoolMethod_MAX) {
              // effective_mask = index_out*4 + mask_ptr; stored in reg_scratch3
              mov(reg_scratch0, stack_top_mask_ptr);
              lea(reg_scratch3, ptr[reg_scratch6*4 + reg_scratch0]);
            }

            // Iterate through output width.
            mov(reg_scratch8, 0);
            L("out_w_loop_start");
            cmp(reg_scratch8, layer->pooled_width_);
            jae("out_w_loop_end", T_NEAR);

              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_MAX) {
                mov(reg_index_cnt, reg_scratch7);
                movaps(xmm0, xmm15);
                for (int kernel_h = 0;
                         kernel_h < layer->kernel_h_; ++kernel_h) {
                  for (int kernel_w = 0;
                           kernel_w < layer->kernel_w_; ++kernel_w) {
                    ucomiss(xmm0,
                        ptr[reg_scratch5
                            + kernel_w*sizeof(float)
                            + kernel_h*layer->width_*sizeof(float)]);
                    maxss(xmm0,
                        ptr[reg_scratch5
                            + kernel_w*sizeof(float)
                            + kernel_h*layer->width_*sizeof(float)]);
                    cmovb(reg_index_acc, reg_index_cnt);

                    if (kernel_w+1 < layer->kernel_w_)
                      inc(reg_index_cnt);
                  }
                  add(reg_index_cnt, layer->width_ - layer->kernel_w_ + 1);
                }

                if (Use_top_mask) {
                  // Use float mask, convert value to float first.
                  cvtsi2ss(xmm1, reg_index_acc);
                  movss(ptr[reg_scratch3], xmm1);
                } else {
                  // Use 32bit integer mask.
                  mov(dword[reg_scratch3], reg_index_acc_l);
                }
              } else if (param.pooling_param().pool()
                         == PoolingParameter_PoolMethod_AVE) {
                xorps(xmm0, xmm0);
                for (int kernel_h = 0;
                     kernel_h < layer->kernel_h_; ++kernel_h)
                  for (int kernel_w = 0;
                      kernel_w < layer->kernel_w_; ++kernel_w)
                    addss(xmm0,
                      ptr[reg_scratch5
                        + kernel_w*sizeof(float)
                        + kernel_h*layer->width_*sizeof(float)]);

                mov(reg_scratch0, layer->kernel_h_*layer->kernel_w_);
                cvtsi2ss(xmm1, reg_scratch0);
                divss(xmm0, xmm1);
              }

              movss(ptr[reg_scratch4], xmm0);

              add(reg_scratch5, layer->stride_w_*sizeof(float));
              add(reg_scratch4, sizeof(float));

              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_MAX) {
                add(reg_scratch7, layer->stride_w_);
                add(reg_scratch3, sizeof(float));
              }

              inc(reg_scratch8);
              jmp("out_w_loop_start", T_NEAR);
            L("out_w_loop_end");
          } else {
            // hstart = ph * stride_h_ - pad_h_; stored in reg_scratch3
            mov(reg_mul_param, layer->stride_h_);
            imul(stack_out_h_cnt);
            sub(reg_mul_result_l, layer->pad_h_);
            mov(reg_scratch3, reg_mul_result_l);

            if (param.pooling_param().pool()
                == PoolingParameter_PoolMethod_AVE) {
              // true_height =
              // min(hstart, height_ - kernel_h_ + pad_h_) + kernel_h_ - hstart;
              // required to compute pooling size
              mov(reg_scratch0, reg_scratch3);
              mov(reg_scratch1, height_kernel_h_pad_h_);
              cmp(reg_scratch0, reg_scratch1);
              cmovg(reg_scratch0, reg_scratch1);
              add(reg_scratch0, layer->kernel_h_);
              sub(reg_scratch0, reg_scratch3);
              mov(stack_true_height, reg_scratch0);
            }

            // int hend
            // = min(hstart, height_ - kernel_h_) + kernel_h_;
            // stored in reg_scratch6
            mov(reg_scratch0, reg_scratch3);
            mov(reg_scratch1, height_kernel_h_);
            cmp(reg_scratch0, reg_scratch1);
            cmovg(reg_scratch0, reg_scratch1);
            add(reg_scratch0, layer->kernel_h_);
            mov(reg_scratch6, reg_scratch0);

            // hstart = max(hstart, 0); stored in reg_scratch3
            mov(reg_scratch0, reg_scratch3);
            mov(reg_scratch1, 0);
            cmp(reg_scratch0, reg_scratch1);
            cmovl(reg_scratch0, reg_scratch1);
            mov(reg_scratch3, reg_scratch0);

            // wstart0 = -pad_w_
            mov(stack_wstart0, -layer->pad_w_);

            // pool_index = (ph * pooled_width_)*4;
            mov(reg_mul_param, layer->pooled_width_ * sizeof(float));
            imul(stack_out_h_cnt);
            mov(stack_pool_index, reg_mul_result_l);

            // index0 = hstart * width_;
            mov(reg_mul_param, layer->width_);
            imul(reg_scratch3);
            mov(stack_index0, reg_mul_result_l);

            // Iterate through output width.
            mov(stack_out_w_cnt, 0);
            L("out_w_loop_start");
            cmp(stack_out_w_cnt, layer->pooled_width_);
            jae("out_w_loop_end", T_NEAR);

              // wend = min(wstart0, width_ - kernel_w_) + kernel_w_;
              // stored in reg_scratch7
              mov(reg_scratch7, stack_wstart0);
              mov(reg_scratch1, width_kernel_w_);
              cmp(reg_scratch7, reg_scratch1);
              cmovg(reg_scratch7, reg_scratch1);
              add(reg_scratch7, layer->kernel_w_);

              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_AVE) {
                // true_width =
                //    min(wstart0, width_ - kernel_w_ + pad_h_)
                //    + kernel_w_ - wstart0;
                // required to compute pooling size
                mov(reg_scratch0, stack_wstart0);
                mov(reg_scratch1, width_kernel_w_pad_w_);
                cmp(reg_scratch0, reg_scratch1);
                cmovg(reg_scratch0, reg_scratch1);
                add(reg_scratch0, layer->kernel_w_);
                sub(reg_scratch0, stack_wstart0);
                mov(stack_true_width, reg_scratch0);
              }

              // wstart = max(wstart0, 0); stored in reg_scratch8
              mov(reg_scratch8, stack_wstart0);
              mov(reg_scratch1, 0);
              cmp(reg_scratch8, reg_scratch1);
              cmovl(reg_scratch8, reg_scratch1);

              // wstart0 += stride_w_
              add(stack_wstart0, layer->stride_w_);

              // num_elements_to_do = wend - wstart; stored in reg_scratch7
              sub(reg_scratch7, reg_scratch8);

              // const int index = index0 + wstart; stored in reg_scratch5
              mov(reg_scratch5, stack_index0);
              add(reg_scratch5, reg_scratch8);

              // const int effective_ptr = (index)*4 + input_ptr;
              // stored in reg_scratch4
              lea(reg_scratch4, ptr[reg_scratch5*4]);
              add(reg_scratch4, stack_bottom_data_ptr);

              // Prepare accumulators.
              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_MAX) {
                movaps(xmm0, xmm15);
                mov(reg_index_acc, -1);
              } else if (param.pooling_param().pool()
                         == PoolingParameter_PoolMethod_AVE) {
                xorps(xmm0, xmm0);
              }

              // Iterate through kernel height.
              // We are reusing mulparam reg here (rax)
              // but we won't be MULing inside.
              mov(reg_mul_param, reg_scratch3);
              align(8);
              L("kern_h_loop_start");
              cmp(reg_mul_param, reg_scratch6);
              jae("kern_h_loop_end", T_NEAR);

                mov(reg_input_ptr, reg_scratch4);
                mov(reg_index_cnt, reg_scratch5);
                lea(reg_scratch2, ptr[reg_scratch5 + reg_scratch7]);

                // Iterate through kernel width.
                align(8);
                L("kern_w_loop_start");
                cmp(reg_index_cnt, reg_scratch2);
                jae("kern_w_loop_end", T_NEAR);

                  if (param.pooling_param().pool()
                      == PoolingParameter_PoolMethod_MAX) {
                    movss(xmm1, ptr[reg_input_ptr]);
                    ucomiss(xmm1, xmm0);
                    maxss(xmm0, xmm1);
                    cmova(reg_index_acc, reg_index_cnt);
                  } else if (param.pooling_param().pool()
                           == PoolingParameter_PoolMethod_AVE) {
                    addss(xmm0, ptr[reg_input_ptr]);
                  }

                  add(reg_input_ptr, sizeof(float));

                  inc(reg_index_cnt);
                  jmp("kern_w_loop_start", T_NEAR);
                L("kern_w_loop_end");

                add(reg_scratch4, layer->width_ * sizeof(float));
                add(reg_scratch5, layer->width_);

                inc(reg_mul_param);
                jmp("kern_h_loop_start", T_NEAR);
              L("kern_h_loop_end");

              // Save accumulators.
              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_AVE) {
                mov(reg_mul_param, stack_true_height);
                imul(stack_true_width);
                cvtsi2ss(xmm1, reg_mul_result_l);
                divss(xmm0, xmm1);
              }

              mov(reg_scratch0, stack_pool_index);
              mov(reg_scratch1, stack_top_data_ptr);
              movss(ptr[reg_scratch0 + reg_scratch1], xmm0);

              if (param.pooling_param().pool()
                  == PoolingParameter_PoolMethod_MAX) {
                if (Use_top_mask) {
                // Use float mask, convert value to float first.
                  cvtsi2ss(xmm0, reg_index_acc);
                  mov(reg_scratch1, stack_top_mask_ptr);
                  movss(ptr[reg_scratch0 + reg_scratch1], xmm0);
                } else {  // Use 32bit integer mask.
                  mov(reg_scratch1, stack_top_mask_ptr);
                  mov(dword[reg_scratch0 + reg_scratch1], reg_index_acc_l);
                }
              }

              // Update pool_index.
              add(stack_pool_index, sizeof(float));

              inc(stack_out_w_cnt);
              jmp("out_w_loop_start", T_NEAR);
            L("out_w_loop_end");
          }

          inc(stack_out_h_cnt);
          jmp("out_h_loop_start", T_NEAR);
        L("out_h_loop_end");

        inc(stack_channel_cnt);
        jmp("channel_loop_start", T_NEAR);
      L("channel_loop_end");

      inc(stack_batch_cnt);
      jmp("batch_loop_start", T_NEAR);
    L("batch_loop_end");

    // Restore r12-r15 and rbx registers.
    pop(rbx); pop(r15); pop(r14); pop(r13); pop(r12);

    add(rsp, stack_qwords * 8);
    pop(rbp);
    ret();

    Callback = getCode<Callback_t>();
  } else {  // Take naive path.
    Callback = &Naive;
  }
}
#endif

template <typename Dtype>
PoolingCodeGeneratorBackward<Dtype>::PoolingCodeGeneratorBackward() {
  Callback = NULL;
}

template <typename Dtype>
PoolingCodeGeneratorBackward<Dtype>::~PoolingCodeGeneratorBackward() {
}

template <typename Dtype>
typename PoolingCodeGeneratorBackward<Dtype>::Callback_t
  PoolingCodeGeneratorBackward<Dtype>::Get_callback(
    PoolingLayer<Dtype>* layer, Blob<Dtype>* top) {
  // Wrapper for lazy initialization.
  // Also check if top shape din't change.
  // TODO: do we need to check all blobs' shapes?
  // In future we may add cache for all already found options.
  // Currently there is only one code for last used shape.
  if (Callback == NULL || top->shape() != layer_output_shape_signature) {
    layer_output_shape_signature = top->shape();
    Create_callback(layer);
  }

  return Callback;
}

template <typename Dtype>
void PoolingCodeGeneratorBackward<Dtype>::Naive(
  const Dtype* top_diff,
  Dtype* bottom_diff,
  int batch_start,
  int batch_end,
  bool use_top_mask,
  const void* mask_ptr,
  PoolingLayer<Dtype>* layer) {
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = static_cast<const Dtype*>(mask_ptr);

  int pooled_fm_size = layer->pooled_height_ * layer->pooled_width_;
  int fm_size = layer->height_ * layer->width_;

  switch (layer->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (!use_top_mask) {
      mask = static_cast<const int*>(mask_ptr);
    }
    bottom_diff += fm_size*layer->channels_*batch_start;
    top_diff += pooled_fm_size*layer->channels_*batch_start;
    if (use_top_mask) {
      top_mask += pooled_fm_size*layer->channels_*batch_start;
    } else {
      mask += pooled_fm_size*layer->channels_*batch_start;
    }
    for (int n = batch_start; n < batch_end; ++n) {
      if (use_top_mask) {
        for (int c = 0; c < layer->channels_; ++c) {
          int index0 = 0;
          for (int ph = 0; ph < layer->pooled_height_; ++ph) {
            int index = index0;
            index0 += layer->pooled_width_;
            for (int pw = 0; pw < layer->pooled_width_; ++pw) {
              const int bottom_index = top_mask[index];
              bottom_diff[bottom_index] += top_diff[index];
              ++index;
            }
          }
          bottom_diff += fm_size;
          top_diff += pooled_fm_size;
          top_mask += pooled_fm_size;
        }
      } else {
        for (int c = 0; c < layer->channels_; ++c) {
          int index0 = 0;
          for (int ph = 0; ph < layer->pooled_height_; ++ph) {
            int index = index0;
            index0 += layer->pooled_width_;
            for (int pw = 0; pw < layer->pooled_width_; ++pw) {
              const int bottom_index = mask[index];
              bottom_diff[bottom_index] += top_diff[index];
              ++index;
            }
          }
          bottom_diff += fm_size;
          top_diff += pooled_fm_size;
          mask += pooled_fm_size;
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:

    bottom_diff += fm_size*layer->channels_*batch_start;
    top_diff += pooled_fm_size*layer->channels_*batch_start;
    // The main loop
    for (int n = batch_start; n < batch_end; ++n) {
      for (int c = 0; c < layer->channels_; ++c) {
        for (int ph = 0; ph < layer->pooled_height_; ++ph) {
          for (int pw = 0; pw < layer->pooled_width_; ++pw) {
            int hstart = ph * layer->stride_h_ - layer->pad_h_;
            int wstart = pw * layer->stride_w_ - layer->pad_w_;
            int hend =
              min(hstart + layer->kernel_h_, layer->height_ + layer->pad_h_);
            int wend =
              min(wstart + layer->kernel_w_, layer->width_ + layer->pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, layer->height_);
            wend = min(wend, layer->width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * layer->width_ + w] +=
                  top_diff[ph * layer->pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += fm_size;
        top_diff += pooled_fm_size;
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingCodeGeneratorBackward<Dtype>::Create_callback(
  PoolingLayer<Dtype>* layer) {
  Callback = &Naive;
}

#if defined __x86_64__ || defined _M_X64
// Here we have specialized versions for supported formats in x64 architectures.
/*
template <>
void PoolingCodeGeneratorBackward<float>::Create_callback(
  PoolingLayer<float>* layer)
{
  using namespace ::Xbyak;
  util::Cpu Current_cpu;
  if (Current_cpu.has(util::Cpu::tAVX2))
  { // AVX2 optimized version.
    const LayerParameter& param = layer->layer_param();

    // It seems we are regenerating the code due to output reshape.
    if (Callback)
      reset();

    Callback = getCode<Callback_t>();
  }
  else
  { // Take naive path.
    Callback = Naive;
  }
}*/

#endif

INSTANTIATE_CLASS(PoolingCodeGeneratorForward);
INSTANTIATE_CLASS(PoolingCodeGeneratorBackward);

}  // namespace caffe

