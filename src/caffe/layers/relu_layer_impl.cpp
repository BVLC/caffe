#include "caffe/layers/relu_layer.hpp"

namespace caffe
{
template <typename Dtype>
ReLUCodeGeneratorForward<Dtype>::ReLUCodeGeneratorForward() 
{
  Callback = NULL; 
}

template <typename Dtype>
ReLUCodeGeneratorForward<Dtype>::~ReLUCodeGeneratorForward() 
{
}

template <typename Dtype>
typename ReLUCodeGeneratorForward<Dtype>::Callback_t* ReLUCodeGeneratorForward<Dtype>::Get_callback(ReLULayer<Dtype>* layer, Blob<Dtype>* top) 
{
  // Wrapper for lazy initialization.
  // Also check if top shape din't change.
  // TODO: do we need to check all blobs' shapes?
  // In future we may add cache for all already found options. 
  // Currently there is only one code for last used shape.
  if(Callback == NULL || top->shape() != layer_output_shape_signature)
  {
    layer_output_shape_signature = top->shape();
    Create_callback(layer);
  }

  return Callback;
}

// Implementation of CodeGenerator classes for ReLU.
template <typename Dtype>
void ReLUCodeGeneratorForward<Dtype>::Naive(
  Dtype* top_data, 
  const Dtype* bottom_data, 
  int count,
  Dtype negative_slope)
{
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

// Generic datatypes - use naive versions.
template <typename Dtype>
void ReLUCodeGeneratorForward<Dtype>::Create_callback(ReLULayer<Dtype>* layer)
{
  Callback = Naive;
}

#if defined __x86_64__ || defined _M_X64
// Here we have specialized versions for supported formats in x64 architectures.
template <>
void ReLUCodeGeneratorForward<float>::Create_callback(ReLULayer<float>* layer)
{
  using namespace ::Xbyak;
  util::Cpu Current_cpu;
  if(Current_cpu.has(util::Cpu::tAVX2))
  { // AVX2 optimized version.
    const LayerParameter& param = layer->layer_param();

    if(Callback) // It seems we are regenerating the code due to output reshape.
      reset();

    // Windows have different calling convention, let's adjust it.
    #if defined _WIN64
      // Save RSI and RDI.
      mov(r12, rsi);
      mov(r13, rdi);

      // Move reg arguments.
      mov(rdi, rcx);
      mov(rsi, rdx);
      mov(rdx, r8);
      vmovss(Xmm(0), Xmm(3));
    #endif

    // Registers used as parameters.
    const Reg64& top_data = rdi;
    const Reg64& bottom_data = rsi;
    const Reg64& count = rdx;
    const Ymm& arg_negative_slope_vec = Ymm(0);
    // Number of accumulators used.
    const unsigned int num_acc = 
      param.relu_param().negative_slope() != 0.0f 
        ? 7
        : 14;

    // Size of simd.
    const unsigned int simd_size = 8;

    // Save XMM6-XMM15 (windows only).
    #if defined _WIN64
    sub(rsp, 10 * sizeof(float) * simd_size/2);
    for(int acc_id = 0; acc_id < 10; ++acc_id)
      vmovups(ptr[rsp + acc_id * simd_size/2 * sizeof(float)], Xmm(acc_id+6));
    #endif

    // Registers used later as values.
    const Ymm& zero_vec = Ymm(15);
    const Xmm& zero_ss = Xmm(15);
    const Ymm& negative_slope_vec = Ymm(14);
    const Xmm& negative_slope_ss = Xmm(14);

    const Reg64& full_acc_iterations = r8;
    const Reg64& single_acc_iterations = r9;
    const Reg64& single_float_iterations = r10;

    // Create vector with zeros only and broadcast the negative slope.
    vxorps(zero_vec, zero_vec, zero_vec);
    vpermps(negative_slope_vec, zero_vec, arg_negative_slope_vec);
    
    // RCX <- size of single full acc block (num_acc * simd_size)
    // RAX <- size of whole block to be processed
    // RDX <- zero
    // RDX:RAX / RCX = RAX <- full acc block iterations
    // RDX:RAX % RCX = RDX <- partial result for next division
    mov(rcx, num_acc*simd_size);
    mov(rax, count);
    xor_(rdx, rdx);
    div(rcx);

    // Save RAX.
    mov(full_acc_iterations, rax);

    // RCX <- size of single acc (simd_size)
    // RAX <- previous partial result (left data)
    // RDX <- zero
    // RDX:RAX / RCX = RAX <- single acc iterations
    // RDX:RAX % RCX = RDX <- single float iterations
    mov(rcx, simd_size);
    mov(rax, rdx);
    xor_(rdx, rdx);
    div(rcx);

    // Save RAX and RDX.
    mov(single_acc_iterations, rax);
    mov(single_float_iterations, rdx);

    // The main loop for full acc.
    L("@@");
    cmp(full_acc_iterations, 0);
    je("@f", T_NEAR);

    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vmaxps(Ymm(acc_id), zero_vec, ptr[bottom_data + acc_id * simd_size * sizeof(float)]);
    // Add this code only if there is meaningful negative slope.
    // If there is, then we break accumulators into two parts where
    // one part computes positive slope and second negative slope
    // and at the end theyre summed up.
    if(param.relu_param().negative_slope() != 0.0f)
    {
      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vminps(Ymm(acc_id+num_acc), zero_vec, ptr[bottom_data + acc_id * simd_size * sizeof(float)]);

      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vmulps(Ymm(acc_id+num_acc), Ymm(acc_id+num_acc), negative_slope_vec);
      
      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vaddps(Ymm(acc_id), Ymm(acc_id), Ymm(acc_id+num_acc));
    }
    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vmovups(ptr[top_data + acc_id * simd_size * sizeof(float)], Ymm(acc_id));

    dec(full_acc_iterations);
    add(bottom_data, simd_size * num_acc * sizeof(float));
    add(top_data, simd_size * num_acc * sizeof(float));
    jmp("@b", T_NEAR);


    // The main loop for single acc.
    // That's basically the same code as previously but with one acc.
    // We could just use lovely parametrized lambda, but no C++11 here.
    // Doh..
    L("@@");
    cmp(single_acc_iterations, 0);
    je("@f", T_NEAR);

    vmaxps(Ymm(0), zero_vec, ptr[bottom_data]);
    if(param.relu_param().negative_slope() != 0.0f)
    {
      vminps(Ymm(1), zero_vec, ptr[bottom_data]);
      vmulps(Ymm(1), Ymm(1), negative_slope_vec);
      vaddps(Ymm(0), Ymm(0), Ymm(1));
    }
    vmovups(ptr[top_data], Ymm(0));

    dec(single_acc_iterations);
    add(bottom_data, simd_size * sizeof(float));
    add(top_data, simd_size * sizeof(float));
    jmp("@b");

    // The main loop for single floats.
    L("@@");
    cmp(single_float_iterations, 0);
    je("@f", T_NEAR);

    vmaxss(Xmm(0), zero_ss, ptr[bottom_data]);
    if(param.relu_param().negative_slope() != 0.0f)
    {
      vminss(Xmm(1), zero_ss, ptr[bottom_data]);
      vmulss(Xmm(1), Xmm(1), negative_slope_ss);
      vaddss(Xmm(0), Xmm(0), Xmm(1));
    }
    vmovss(ptr[top_data], Xmm(0));

    dec(single_float_iterations);
    add(bottom_data, sizeof(float));
    add(top_data, sizeof(float));
    jmp("@b", T_NEAR);
    L("@@");

    // Restore XMM6-XMM15 and RSI/RDI(windows only).
    #if defined _WIN64
      for(int acc_id = 0; acc_id < 10; ++acc_id)
        vmovups(Xmm(acc_id+6), ptr[rsp + acc_id * simd_size/2 * sizeof(float)]);
      add(rsp, 10 * sizeof(float) * simd_size/2);

      mov(rsi, r12);
      mov(rdi, r13);
    #endif

    ret();

    Callback = getCode<Callback_t*>();
  }
  else
  { // Take naive path.
    Callback = Naive;
  }
}

#endif

template <typename Dtype>
ReLUCodeGeneratorBackward<Dtype>::ReLUCodeGeneratorBackward() 
{
  Callback = NULL; 
}

template <typename Dtype>
ReLUCodeGeneratorBackward<Dtype>::~ReLUCodeGeneratorBackward() 
{
}

template <typename Dtype>
typename ReLUCodeGeneratorBackward<Dtype>::Callback_t* ReLUCodeGeneratorBackward<Dtype>::Get_callback(ReLULayer<Dtype>* layer, Blob<Dtype>* top) 
{
  // Wrapper for lazy initialization.
  // Also check if top shape din't change.
  // TODO: do we need to check all blobs' shapes?
  // In future we may add cache for all already found options. 
  // Currently there is only one code for last used shape.
  if(Callback == NULL || top->shape() != layer_output_shape_signature)
  {
    layer_output_shape_signature = top->shape();
    Create_callback(layer);
  }

  return Callback;
}

template <typename Dtype>
void ReLUCodeGeneratorBackward<Dtype>::Naive(
  const Dtype* top_diff, 
  Dtype* bottom_diff, 
  const Dtype* bottom_data, 
  int count, 
  Dtype negative_slope)
{
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
        + negative_slope * (bottom_data[i] <= 0));
  }
}

template <typename Dtype>
void ReLUCodeGeneratorBackward<Dtype>::Create_callback(ReLULayer<Dtype>* layer)
{
  Callback = Naive;
}

#if defined __x86_64__ || defined _M_X64
// Here we have specialized versions for supported formats in x64 architectures.
template <>
void ReLUCodeGeneratorBackward<float>::Create_callback(ReLULayer<float>* layer)
{
  using namespace ::Xbyak;
  util::Cpu Current_cpu;
  if(Current_cpu.has(util::Cpu::tAVX2))
  { // AVX2 optimized version.
    const LayerParameter& param = layer->layer_param();

    // Size of simd.
    const unsigned int simd_size = 8;

    if(Callback) // It seems we are regenerating the code due to output reshape.
      reset();

    // Windows have different calling convention, let's adjust it.
    #if defined _WIN64
      // Save RSI and RDI.
      mov(r12, rsi);
      mov(r13, rdi);

      // Move reg arguments.
      mov(rdi, rcx);
      mov(rsi, rdx);
      mov(rdx, r8);
      mov(rcx, r9);

      // Fifth argument will be on stack in Windows version.
      const unsigned int shadow_space_offset = 32;
      vmovss(Xmm(0), ptr[esp+shadow_space_offset+sizeof(float)]);

      // Save XMM6-XMM15.
      sub(rsp, 10 * sizeof(float) * simd_size/2);
      for(int acc_id = 0; acc_id < 10; ++acc_id)
        vmovups(ptr[rsp + acc_id * simd_size/2 * sizeof(float)], Xmm(acc_id+6));

    #endif

    // Registers used as parameters.
    const Reg64& top_diff = rdi;
    const Reg64& bottom_diff = rsi;
    const Reg64& bottom_data_arg = rdx;
    const Reg64& count = rcx;
    const Ymm& arg_negative_slope_vec = Ymm(0);

    // Number of accumulators used.
    const unsigned int num_acc = 
      param.relu_param().negative_slope() != 0.0f 
        ? 6
        : 12;

    // Registers used later as values.
    const Ymm& zero_vec = Ymm(15);
    const Xmm& zero_ss = Xmm(15);
    const Ymm& negative_slope_vec = Ymm(14);
    const Xmm& negative_slope_ss = Xmm(14);
    const Ymm& one_vec = Ymm(13);
    const Xmm& one_ss = Xmm(13);

    mov(rax, 0x3f800000);
    push(rax);
    vbroadcastss(one_vec, ptr[rsp]);
    pop(rax);

    const Reg64& full_acc_iterations = r8;
    const Reg64& single_acc_iterations = r9;
    const Reg64& single_float_iterations = r10;

    // Move RDX to different register as it will be overrided by DIVisions.
    const Reg64& bottom_data = r11;
    mov(bottom_data, bottom_data_arg);

    // Create vector with zeros only and broadcast the negative slope.
    vxorps(zero_vec, zero_vec, zero_vec);
    vpermps(negative_slope_vec, zero_vec, arg_negative_slope_vec);

    // RCX <- size of single full acc block (num_acc * simd_size)
    // RAX <- size of whole block to be processed
    // RDX <- zero
    // RDX:RAX / RCX = RAX <- full acc block iterations
    // RDX:RAX % RCX = RDX <- partial result for next division
    mov(rax, count);
    mov(rcx, num_acc*simd_size);
    xor_(rdx, rdx);
    div(rcx);

    // Save RAX.
    mov(full_acc_iterations, rax);

    // RCX <- size of single acc (simd_size)
    // RAX <- previous partial result (left data)
    // RDX <- zero
    // RDX:RAX / RCX = RAX <- single acc iterations
    // RDX:RAX % RCX = RDX <- single float iterations
    mov(rcx, simd_size);
    mov(rax, rdx);
    xor_(rdx, rdx);
    div(rcx);

    // Save RAX and RDX.
    mov(single_acc_iterations, rax);
    mov(single_float_iterations, rdx);

    // The main loop for full acc.
    L("@@");
    cmp(full_acc_iterations, 0);
    je("@f", T_NEAR);

    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vcmpltps(Ymm(acc_id), zero_vec, ptr[bottom_data + acc_id * simd_size * sizeof(float)]);

    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vpand(Ymm(acc_id), Ymm(acc_id), one_vec);
    // Add this code only if there is meaningful negative slope.
    // If there is, then we break accumulators into two parts where
    // one part computes positive slope and second negative slope
    // and at the end theyre summed up.
    if(param.relu_param().negative_slope() != 0.0f)
    {
      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vcmpgeps(Ymm(acc_id+num_acc), zero_vec, ptr[bottom_data + acc_id * simd_size * sizeof(float)]);

      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vpand(Ymm(acc_id+num_acc), Ymm(acc_id+num_acc), negative_slope_vec);

      for(int acc_id = 0; acc_id < num_acc; ++acc_id)
        vaddps(Ymm(acc_id), Ymm(acc_id), Ymm(acc_id+num_acc));
    }

    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vmulps(Ymm(acc_id), Ymm(acc_id), ptr[top_diff + acc_id * simd_size * sizeof(float)]);
    
    for(int acc_id = 0; acc_id < num_acc; ++acc_id)
      vmovups(ptr[bottom_diff + acc_id * simd_size * sizeof(float)], Ymm(acc_id));


    dec(full_acc_iterations);
    add(bottom_data, simd_size * num_acc * sizeof(float));
    add(bottom_diff, simd_size * num_acc * sizeof(float));
    add(top_diff, simd_size * num_acc * sizeof(float));
    jmp("@b", T_NEAR);


    // The main loop for single acc.
    // That's basically the same code as previously but with one acc.
    // We could just use lovely parametrized lambda, but no C++11 here.
    // Doh..
    L("@@");
    cmp(single_acc_iterations, 0);
    je("@f", T_NEAR);

    vcmpltps(Ymm(0), zero_vec, ptr[bottom_data]);
    vpand(Ymm(0), Ymm(0), one_vec);
    if(param.relu_param().negative_slope() != 0.0f)
    {
      vcmpgeps(Ymm(1), zero_vec, ptr[bottom_data]);
      vpand(Ymm(1), Ymm(1), negative_slope_vec);
      vaddps(Ymm(0), Ymm(0), Ymm(1));
    }
    vmulps(Ymm(0), Ymm(0), ptr[top_diff]);
    vmovups(ptr[bottom_diff], Ymm(0));

    dec(single_acc_iterations);
    add(bottom_data, simd_size * sizeof(float));
    add(bottom_diff, simd_size * sizeof(float));
    add(top_diff, simd_size * sizeof(float));
    jmp("@b");

    // The main loop for single floats.
    L("@@");
    cmp(single_float_iterations, 0);
    je("@f", T_NEAR);
    
    vcmpltss(Xmm(0), zero_ss, ptr[bottom_data]);
    vpand(Xmm(0), Xmm(0), one_ss);
    if(param.relu_param().negative_slope() != 0.0f)
    {
      vcmpgess(Xmm(1), zero_ss, ptr[bottom_data]);
      vpand(Xmm(1), Xmm(1), negative_slope_ss);
      vaddss(Xmm(0), Xmm(0), Xmm(1));
    }
    vmulss(Xmm(0), Xmm(0), ptr[top_diff]);
    vmovss(ptr[bottom_diff], Xmm(0));
    
    dec(single_float_iterations);
    add(bottom_data, sizeof(float));
    add(bottom_diff, sizeof(float));
    add(top_diff, sizeof(float));
    jmp("@b", T_NEAR);
    L("@@");

    // Restore XMM6-XMM15 and RSI/RDI(windows only).
    #if defined _WIN64
      for(int acc_id = 0; acc_id < 10; ++acc_id)
        vmovups(Xmm(acc_id+6), ptr[rsp + acc_id * simd_size/2 * sizeof(float)]);
      add(rsp, 10 * sizeof(float) * simd_size/2);

      mov(rsi, r12);
      mov(rdi, r13);
    #endif

    ret();

    Callback = getCode<Callback_t*>();
  }
  else
  { // Take naive path.
    Callback = Naive;
  }
}

#endif

INSTANTIATE_CLASS(ReLUCodeGeneratorForward);
INSTANTIATE_CLASS(ReLUCodeGeneratorBackward);

}