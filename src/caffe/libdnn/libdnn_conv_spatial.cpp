#include <string>
#include <vector>
#include "caffe/common.hpp"
#ifdef USE_INTEL_SPATIAL
#include "caffe/backend/device.hpp"
#include "caffe/util/benchmark.hpp"

// #define LIBDNN_DEBUG 1
#ifdef USE_OPENCL
#include <boost/filesystem.hpp>
#include "caffe/greentea/cl_kernels.hpp"
#include "viennacl/tools/sha1.hpp"
// #define TEST_ALL_KERNELS
namespace caffe {

#define ALIGN(val, N) (((val) + (N) - 1) & ~((N) - 1))

template<typename MItype>
LibDNNConvSpatial<MItype>::LibDNNConvSpatial(LibDNNConvConfig config) {
  config_ = config;

  // Initialize the sidekick kernels for backward pass
  LibDNNConv<MItype>* libdnn_conv = new LibDNNConv<MItype>(config);
  libdnn_conv_.reset(libdnn_conv);

  LibDNN<MItype>::dev_ptr_ = config.dev_ptr;
  this->bias_term_ = config.bias_term;
  this->bias_multiplier_ = config.bias_term ? 1.0 : 0.0;
  LibDNN<MItype>::fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.in_shape.size();
  int_tp spatial_dims = config.kernel.size();

  this->num_axes_ = spatial_dims;
  this->fmaps_in_ = config.in_shape[dims - spatial_dims - 1];
  this->fmaps_out_ = config.out_shape[dims - spatial_dims - 1];

  this->group_ = config.group;

  for (int_tp i = 0; i < spatial_dims; ++i) {
    this->kernel_shape_.push_back(config.kernel[i]);
    this->pad_.push_back(config.pad[i]);
    this->stride_.push_back(config.stride[i]);
    this->dilation_.push_back(config.dilation[i]);
    this->im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
    this->im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
  }

  bias_ = NULL;
  tuned_ = false;
  try_cache_ = false;
  swizzled_weights_ = NULL;
  kernel_dim_ = this->fmaps_in_ / this->group_;
  in_spatial_dim_ = 1;
  out_spatial_dim_ = 1;
  for (int_tp i = 0; i < spatial_dims; ++i) {
     kernel_dim_ *= config.kernel[i];
     in_spatial_dim_ *= config.in_shape[dims - spatial_dims + i];
     out_spatial_dim_ *= config.out_shape[dims - spatial_dims + i];
  }

  is_1x1_ = true;
  for (int_tp i = 0; i < spatial_dims; ++i) {
     is_1x1_ &= this->kernel_shape_[i] == 1
             && this->stride_[i] == 1
             && this->pad_[i] == 0;
     if (!is_1x1_) {
        break;
     }
  }

  this->M_FW_ = this->fmaps_out_ / this->group_;
  this->K_FW_ = this->fmaps_in_ * this->kernel_shape_[0]
                                * this->kernel_shape_[1] / this->group_;

  height_ = this->im_in_shape_[0];
  width_ = this->im_in_shape_[1];
  const int_tp kernel_extent_h = this->dilation_[0]
                          * (this->kernel_shape_[0] - 1) + 1;
  const int_tp kernel_extent_w = this->dilation_[1]
                          * (this->kernel_shape_[1] - 1) + 1;
  output_h_ = (height_ + 2 * this->pad_[0] - kernel_extent_h)
                          / this->stride_[0] + 1;
  output_w_ = (width_ + 2 * this->pad_[1] - kernel_extent_w)
                          / this->stride_[1] + 1;

  bottom_dim_ = this->fmaps_in_ * in_spatial_dim_;
  top_dim_ = this->fmaps_out_ * out_spatial_dim_;

  GenerateHelperKernels();
  LibDNN<MItype>::CompileKernels();

  if (std::getenv("CLCAFFE_CACHE_PATH"))
    cache_path_ << std::getenv("CLCAFFE_CACHE_PATH");
  else if (std::getenv("VIENNACL_CACHE_PATH"))
    cache_path_ << std::getenv("VIENNACL_CACHE_PATH") << "/clCaffe";
  else if (std::getenv("HOME")) {
    cache_path_ << std::getenv("HOME") << "/.cache/clCaffe";
  }
  cache_path_ << "/spatialkernels/";
  const boost::filesystem::path& path = cache_path_.str();
  const boost::filesystem::path& dir =
                 boost::filesystem::unique_path(path).string();
  bool hasCacheDir = false;
  if (!boost::filesystem::exists(dir))
    hasCacheDir = boost::filesystem::create_directories(dir);
  else
    hasCacheDir = boost::filesystem::is_directory(dir);

  if (hasCacheDir != true) {
    std::cout << "Failed to create cache directory,"
              << "will tune again for next running" << std::endl;
    return;
  }
}

template<typename MItype>
string LibDNNConvSpatial<MItype>::generate_fw_defs() {
  stringstream ss;

  ss << "#define __CAT(X, Y) X##Y" << std::endl;
  ss << "#define CAT(X, Y) __CAT(X, Y)" << std::endl;
  ss << "#define LOOP0(VAR, STMT)" << std::endl;
  ss << "#define LOOP1(VAR, STMT) (STMT); (VAR)++;" << std::endl;
  ss << "#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;"
     << std::endl;
  ss << "#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))"
     << std::endl;

  ss << this->program_->define("KERNEL_WIDTH", this->kernel_shape_[1]);
  ss << this->program_->define("KERNEL_HEIGHT" , this->kernel_shape_[0]);
  ss << this->program_->define("STRIDE_X", this->stride_[1]);
  ss << this->program_->define("STRIDE_Y", this->stride_[0]);
  ss << this->program_->define("DILATION_X", this->dilation_[1]);
  ss << this->program_->define("DILATION_Y", this->dilation_[0]);
  ss << this->program_->define("INPUT_PAD_W", this->pad_[1]);
  ss << this->program_->define("INPUT_PAD_H", this->pad_[0]);

  return ss.str();
}

typedef enum {
  KERNEL_TYPE_INTEL_IDLF = 2,
  KERNEL_TYPE_BASIC = 4,
  KERNEL_TYPE_GEMM_LIKE = 5
} libdnnConvSpatialKernelType_t;

template<typename MItype>
string LibDNNConvSpatial<MItype>::generate_fw_kernels(int_tp kernelType,
                                                          int_tp blockM,
                                                          int_tp blockK,
                                                          int_tp blockN) {
  stringstream ss;
  stringstream opts;
  string kernelUKey;
  int_tp simd_size;
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());

  if (kernelType == KERNEL_TYPE_INTEL_IDLF) {
    simd_size = blockN;
    kernelUKey = generate_specific_key(2, blockM, blockK, 1);

    // kernel name
    kernel_name_ = "IDLF_";
    kernel_name_ += kernelUKey.c_str();
    if (simd_size == 16)
       kernel_name_ += "_SIMD16";
    else
       kernel_name_ += "_SIMD8";

    // options
    opts << "-cl-fast-relaxed-math -D convolve_simd=" << kernel_name_;
    if (IsBeignet(&ctx))
      opts << " -D__BEIGNET__ ";
    options_ = opts.str();

    // defs
    int_tp output_width = output_w_;
    int_tp output_height = output_h_;
    int_tp output_block_width = blockM;
    int_tp output_block_height = blockK;
    const int_tp last_block_width =
       (output_width % output_block_width == 0) ?
       output_block_width : output_width % output_block_width;
    const int_tp last_block_height =
       (output_height % output_block_height == 0) ?
       output_block_height : output_height % output_block_height;
    int_tp tile_x = (((output_block_width - 1) * this->stride_[1]
             + this->kernel_shape_[1] * this->dilation_[1]) + 3) & ~3;
    int_tp tile_y = (output_block_height -1)
             * this->stride_[0] + this->kernel_shape_[0] * this->dilation_[0];
    int_tp tile_y_stride = (4 * simd_size) / tile_x;
    int_tp invec_size = (tile_y + tile_y_stride - 1) / tile_y_stride;

    ss << this->program_->define("SIMD_SIZE", simd_size);
    ss << this->program_->define("filter_qualifier", "__global");
    ss << this->program_->define("OUT_BLOCK_WIDTH", output_block_width);
    ss << this->program_->define("OUT_BLOCK_HEIGHT", output_block_height);
    ss << this->program_->define("LAST_BLOCK_WIDTH", last_block_width);
    ss << this->program_->define("LAST_BLOCK_HEIGHT", last_block_height);
    ss << this->program_->define("INPUT_DEPTH", this->fmaps_in_ / this->group_);
    ss << this->program_->define("TOTAL_INPUT_DEPTH_SIZE", this->fmaps_in_);
    ss << this->program_->define("TOTAL_OUTPUT_DEPTH", this->fmaps_out_);
    ss << this->program_->define("INPUT_START_X", 0);
    ss << this->program_->define("INPUT_START_Y", 0);
    ss << this->program_->define("INPUT_START_Z", 0);
    ss << this->program_->define("NUM_FILTERS", this->M_FW_);
    ss << this->program_->define("OUT_BUFF_OFFSET", 0);
    ss << this->program_->define("TILE_X", tile_x);
    ss << this->program_->define("TILE_Y", tile_y);
    ss << this->program_->define("TILE_Y_STRIDE", tile_y_stride);
    ss << this->program_->define("INVEC_SIZE", invec_size);
    ss << this->program_->define("ALIGNED_NUM_FILTERS",
                           ALIGN(this->M_FW_, simd_size));
    ss << this->program_->define("OUT_BLOCK_SIZE",
          (output_block_width*output_block_height));

    // kernel source
    // Each work-item computes
    // a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.
    // Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread)
    // will compute 16/8 different feature maps,
    // but each feature map is for the same region of the imput image.
    // NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH,
    //           (output_height+pad)/OUT_BLOCK_HEIGHT,
    //           NUM_FILTERS/OUT_BLOCK_DEPTH
    // NOTE: for beignet
    // this reqd_work_group_size does not guarantee that
    // SIMD16/8 mode will be used,
    // the compiler could choose to use two SIMD8 threads,
    // and if that happens the code will break.
    ss << "#if defined(convolve_simd) || defined(Conv_Interleaved)"
       << std::endl;
    ss << "#if TYPE == TYPE_HALF" << std::endl;
    ss << "#define INT_TYPE ushort" << std::endl;
    ss << "#define INT_TYPE2 ushort2" << std::endl;
    ss << "#define INT_TYPE4 ushort4" << std::endl;
    ss << "#define INT_TYPE8 ushort8" << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read_us2"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read_us4"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read_us8"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read_us"
       << std::endl;
    ss << "#else" << std::endl;
    ss << "#define INT_TYPE uint" << std::endl;
    ss << "#define INT_TYPE2 uint2" << std::endl;
    ss << "#define INT_TYPE4 uint4" << std::endl;
    ss << "#define INT_TYPE8 uint8" << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read2"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read4"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read8"
       << std::endl;
    ss << "#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read"
       << std::endl;
    ss << "#endif" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#define activation_function(X) (X)" << std::endl;
    ss << "__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))"
       << std::endl;
    ss << "kernel void" << std::endl;
    ss << "convolve_simd(" << std::endl;
    ss << "__global MItype* inputs_base," << std::endl;
    ss << "filter_qualifier MItype* weights_base," << std::endl;
    ss << "__global MItype* biases_base," << std::endl;
    ss << "__global MItype* outputs_base," << std::endl;
    ss << "const ushort input_width," << std::endl;
    ss << "const ushort input_height," << std::endl;
    ss << "const ushort output_width," << std::endl;
    ss << "const ushort output_height)" << std::endl;
    ss << "{" << std::endl;
    ss << "__global MItype* outputs = outputs_base;" << std::endl;
    ss << "__global MItype* inputs = inputs_base;" << std::endl;
    ss << "filter_qualifier MItype* weights = weights_base;" << std::endl;
    ss << "__global MItype* biases = biases_base;" << std::endl;
    // oc = Output Column
    ss << "uint_tp oc = get_global_id(0) * OUT_BLOCK_WIDTH;" << std::endl;
    // or = Output Row
    ss << "uint_tp or = get_global_id(1) * OUT_BLOCK_HEIGHT;" << std::endl;
    // fm = Feature Map = od = Output Depth
    ss << "uint_tp fm = get_global_id(2);" << std::endl;
    ss << "uint_tp fmg = get_group_id(2);" << std::endl;
    ss << "uint_tp lid = get_local_id(2);" << std::endl;
    ss << "MItype out[OUT_BLOCK_SIZE];" << std::endl;
    ss << "int_tp in_addr;" << std::endl;
    // find weights adress of given neuron (lid is index)
    ss << "uint_tp weight_addr = (fmg % (ALIGNED_NUM_FILTERS/SIMD_SIZE)) * "
       << "INPUT_DEPTH * KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE + lid;"
       << std::endl;
    ss << "for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {" << std::endl;
    ss << "out[i]=0.0f;" << std::endl;
    ss << "}" << std::endl;
    ss << "uint_tp num_in_batch = ( fm ) / ALIGNED_NUM_FILTERS;" << std::endl;
    ss << "uint_tp input_batch_offset = "
       << "num_in_batch * input_height * input_width * TOTAL_INPUT_DEPTH_SIZE;"
       << std::endl;
    ss << "int_tp curr_y = or * STRIDE_Y"
       << " + INPUT_START_Y + (lid / (TILE_X/4));"
       << std::endl;
    ss << "int_tp curr_x = oc * STRIDE_X"
       << " + INPUT_START_X + (lid % (TILE_X/4)) * 4;"
       << std::endl;
    ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
    ss << "int_tp saved_y = curr_y;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "in_addr = "
       << "input_batch_offset + INPUT_START_Z * input_height * input_width"
       // Y tile offset
       << "+  (curr_y - INPUT_PAD_H) * input_width"
       // X tile offset
       << "+   curr_x - INPUT_PAD_W;"
       << std::endl;
    ss << "union {" << std::endl;
    ss << "Dtype4 in_vec[INVEC_SIZE];" << std::endl;
    ss << "MItype in_array[INVEC_SIZE * 4];" << std::endl;
    ss << "} in_buf;" << std::endl;
    ss << "for(int_tp kd = 0; kd < INPUT_DEPTH; kd++)" << std::endl;
    ss << "{" << std::endl;
    ss << "int_tp in_offset = in_addr;" << std::endl;
    ss << "int_tp reg = 0;" << std::endl;
    ss << "LOOP(INVEC_SIZE, reg," << std::endl;
    ss << "{" << std::endl;
    ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
    ss << "if (curr_y >= INPUT_PAD_H && "
       << "curr_y < input_height + INPUT_PAD_H && "
       << "curr_x + 3 >= INPUT_PAD_W && "
       << "curr_x < input_width + INPUT_PAD_W) {" << std::endl;
    ss << "if (curr_x < INPUT_PAD_W) {" << std::endl;
    ss << "in_buf.in_vec[reg].s0 = 0;" << std::endl;
    ss << "if (curr_x + 1 >= INPUT_PAD_W)" << std::endl;
    ss << "in_buf.in_vec[reg].s1 = *(inputs + in_offset + 1);" << std::endl;
    ss << "else" << std::endl;
    ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
    ss << "if (curr_x + 2 >= INPUT_PAD_W)" << std::endl;
    ss << "in_buf.in_vec[reg].s2 = *(inputs + in_offset + 2);" << std::endl;
    ss << "else" << std::endl;
    ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
    ss << "in_buf.in_vec[reg].s3 = *(inputs + in_offset + 3);" << std::endl;
    ss << "} else {" << std::endl;
    // read SIMD_SIZE elements
    ss << "in_buf.in_vec[reg] = vload4(0, inputs + in_offset);"
       << std::endl;
    ss << "if (curr_x + 1 >= input_width + INPUT_PAD_W)" << std::endl;
    ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
    ss << "if (curr_x + 2 >= input_width + INPUT_PAD_W)" << std::endl;
    ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
    ss << "if (curr_x + 3 >= input_width + INPUT_PAD_W)" << std::endl;
    ss << "in_buf.in_vec[reg].s3 = 0;" << std::endl;
    ss << "}" << std::endl;
    ss << "} else {" << std::endl;
    ss << "in_buf.in_vec[reg] = 0;" << std::endl;
    ss << "}" << std::endl;
    ss << "curr_y += TILE_Y_STRIDE;" << std::endl;
    ss << "#else" << std::endl;
    // read SIMD_SIZE elements
    ss << "in_buf.in_vec[reg] = *(global Dtype4*)(inputs + in_offset);"
       << std::endl;
    ss << "#endif" << std::endl;
    ss << "in_offset += input_width * TILE_Y_STRIDE;" << std::endl;
    ss << "});" << std::endl;
    ss << "in_addr += input_height * input_width;" << std::endl;
    ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
    ss << "curr_y = saved_y;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
    ss << "#define WEIGHT_PREF 8" << std::endl;
    ss << "#else" << std::endl;
    ss << "#define WEIGHT_PREF 1" << std::endl;
    ss << "#endif" << std::endl;
    ss << "union {" << std::endl;
    ss << "MItype w[WEIGHT_PREF];" << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
    ss << "INT_TYPE8 ui8;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "} weight_buf;" << std::endl;
    ss << "int_tp w_idx=0;" << std::endl;
    ss << "uint_tp orig_weight_addr = weight_addr;" << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
    ss << "weight_buf.ui8 = "
       << "SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);"
       << std::endl;
    ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
    ss << "#else" << std::endl;
    ss << "weight_buf.w[0] = as_Dtype("
       << "SUB_GROUP_BLOCK_READ((__global INT_TYPE *)&weights[weight_addr]));"
       << std::endl;
    ss << "weight_addr += SIMD_SIZE * 1;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#define BLOCK_IN(N) "
       << "sub_group_broadcast("
       << "in_buf.in_array[((N)%4) + ((N) / (TILE_Y_STRIDE * TILE_X)) * 4], "
       << "(((N) % (TILE_Y_STRIDE * TILE_X))/4))" << std::endl;
    // kr = Kernel Row
    ss << "int_tp kr = 0;" << std::endl;
    ss << "LOOP(KERNEL_HEIGHT, kr," << std::endl;
    ss << "{" << std::endl;
    // kc = Kernel Column
    ss << "int_tp kc = 0;" << std::endl;
    ss << "LOOP(KERNEL_WIDTH, kc," << std::endl;
    ss << "{" << std::endl;
    ss << "for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {" << std::endl;
    ss << "for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {" << std::endl;
    ss << "MItype input = BLOCK_IN((br * STRIDE_Y + kr * DILATION_Y) * "
       << "TILE_X + bc * STRIDE_X + kc * DILATION_X);" << std::endl;
    ss << "out[br * OUT_BLOCK_WIDTH + bc] = "
       << "mad(weight_buf.w[w_idx % WEIGHT_PREF], "
       << "input, out[br * OUT_BLOCK_WIDTH + bc]);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT > WEIGHT_PREF" << std::endl;
    // We assume KERNEL_WIDTH is equal to KERNEL_HEIGHT here.
    ss << "if ((w_idx + 1) % WEIGHT_PREF == 0" << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 != 0" << std::endl;
    ss << "&& ((w_idx + 1) <= (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF))"
       << std::endl;
    ss << "#endif" << std::endl;
    ss << ") {" << std::endl;
    ss << "weight_buf.ui8 = "
       << "SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);"
       << std::endl;
    // weights must be stored in just the right SIMD swizzled format
    // for this to work, see host code for details.
    ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
    ss << "}" << std::endl;
    ss << "#if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0" << std::endl;
    // need to do nothing
    ss << "#else" << std::endl;
    ss << "else if ((w_idx + 1) %  WEIGHT_PREF == 0 && "
       << "((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))"
       << std::endl;
    ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1" << std::endl;
    ss << "weight_buf.w[0] = weights[weight_addr];" << std::endl;
    ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2" << std::endl;
    ss << "weight_buf.ui8.s01 = "
       << "SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)&weights[weight_addr]);"
       << std::endl;
    ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4" << std::endl;
    ss << "weight_buf.ui8.s0123 = "
       << "SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)&weights[weight_addr]);"
       << std::endl;
    ss << "#else" << std::endl;
    ss << "weight_buf.ui8 = "
       << "SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);"
       << std::endl;
    ss << "#endif" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#endif" << std::endl;
    ss << "++w_idx;" << std::endl;
    ss << "});" << std::endl;
    ss << "});" << std::endl;
    ss << "weight_addr = "
       << "orig_weight_addr + KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE;"
       << std::endl;
    ss << "}" << std::endl;
    // dead code to work around possible compiler bug.
    ss << "if (ALIGNED_NUM_FILTERS != NUM_FILTERS && fm > 0xfffffffeul) {"
       << std::endl;
    ss << "outputs[0] = BLOCK_IN(fm % SIMD_SIZE);" << std::endl;
    ss << "}" << std::endl;
    ss << "fm = fm % ALIGNED_NUM_FILTERS;" << std::endl;
    ss << "if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {"
       << std::endl;
    ss << "uint_tp out_addr = "
       << "OUT_BUFF_OFFSET + "
       << "( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * "
       << "output_width * output_height;"
       << std::endl;
    ss << "out_addr += or * output_width + oc;" << std::endl;
    ss << "MItype bias = biases[(fm % ALIGNED_NUM_FILTERS)];" << std::endl;
    ss << "#ifndef WRITE_PADDED_VALUES" << std::endl;
    ss << "if(get_global_id(0) != (get_global_size(0)-1) &&" << std::endl;
    ss << "get_global_id(1) != (get_global_size(1)-1) )" << std::endl;
    ss << "{" << std::endl;
    ss << "#endif" << std::endl;
    ss << "for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {" << std::endl;
    ss << "for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {" << std::endl;
    // this does a scattered write to SIMD_SIZE different feature maps,
    // so that data within one map is contiguous,
    // thus ready for input to next layer.
    ss << "outputs[out_addr + r * output_width + c] = "
       << "activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "#ifndef WRITE_PADDED_VALUES" << std::endl;
    ss << "} else if ( get_global_id(1) != (get_global_size(1)-1) )"
       << std::endl;
    ss << "{" << std::endl;
    ss << "for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {" << std::endl;
    ss << "for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {" << std::endl;
    ss << "outputs[out_addr + r * output_width + c] = "
       << "activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "else if ( get_global_id(0) != (get_global_size(0)-1) )" << std::endl;
    ss << "{" << std::endl;
    ss << "for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {" << std::endl;
    ss << "for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {" << std::endl;
    ss << "outputs[out_addr + r * output_width + c] = "
       << "activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "else" << std::endl;
    ss << "{" << std::endl;
    ss << "for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {" << std::endl;
    ss << "for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {" << std::endl;
    ss << "outputs[out_addr + r * output_width + c] = "
       << "activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "#endif" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else if (kernelType == KERNEL_TYPE_GEMM_LIKE) {
    simd_size = blockK;
    kernelUKey = generate_specific_key(kernelType, blockM, blockK, blockN);
    // kernel name
    kernel_name_ = "U_GEMM_LIKE_CONV_";
    kernel_name_ += kernelUKey.c_str();
    if (simd_size == 8)
      kernel_name_ += "_SIMD8";
    else
      kernel_name_ += "_SIMD16";

    // kernel specific options
    stringstream kernelDef;
    kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
    if (simd_size == 8) {
      kernelDef << "_SIMD8";
    } else {
      kernelDef << "_SIMD16";
    }
    opts << "-cl-fast-relaxed-math -cl-mad-enable -D "
         << kernelDef.str() << " -D Conv_Interleaved="
         << kernel_name_.c_str();
    if (IsBeignet(&ctx))
      opts << " -D__BEIGNET__";
    else
      opts << " -cl-no-subgroup-ifp ";
    options_ = opts.str();

    int_tp tile_n_last_div8 = (this->M_FW_ % 32) / 8;
    ss << this->program_->define("INPUT_DEPTH", this->fmaps_in_);
    ss << this->program_->define("WIDTH1", this->M_FW_);
    ss << this->program_->define("OUT_PADDING_LEFT", 0);
    ss << this->program_->define("OUT_PADDING_HEIGHT", 0);
    ss << this->program_->define("OUT_DEPTH", this->M_FW_);
    ss << this->program_->define("KERNEL_WIDTH_DIV2", this->kernel_shape_[1] / 2);
    ss << this->program_->define("KERNEL_SLICE_DIV2", (this->kernel_shape_[1]
                                                   * this->kernel_shape_[0])/2);
    ss << this->program_->define("TILE_N_LAST", this->M_FW_ % 32);
    ss << this->program_->define("TILE_N_LAST_DIV8", tile_n_last_div8);
    ss << this->program_->define("TILE_M", blockM);
    ss << this->program_->define("TILE_N_PER_LANE", 32 / simd_size);

#define TYPEDEF_FLOAT_N(ele_num) \
        do { \
        ss << "typedef struct MItype" << ele_num << " { "; \
        for (int_tp i = 0; i < ele_num; i++) { ss << "MItype s" << i << "; ";} \
        ss << "} MItype" << ele_num << ";" << std::endl; \
        } while (0)

    TYPEDEF_FLOAT_N(1);
    TYPEDEF_FLOAT_N(5);
    TYPEDEF_FLOAT_N(6);
    TYPEDEF_FLOAT_N(7);
    TYPEDEF_FLOAT_N(9);
    TYPEDEF_FLOAT_N(10);
    TYPEDEF_FLOAT_N(11);
    TYPEDEF_FLOAT_N(12);
    TYPEDEF_FLOAT_N(13);
    TYPEDEF_FLOAT_N(14);
    TYPEDEF_FLOAT_N(15);
    // never used but makes compiler happy.
    ss << "typedef struct Dtype0 { MItype s0; } Dtype0;" << std::endl;

    ss << this->program_->define("OUT_PITCH_X", "output_width");
    ss << this->program_->define("OUT_PITCH_Y", "(output_width * output_height)");
    ss << this->program_->define("ROW_PITCH", "input_width");
    ss << this->program_->define("SLICE_PITCH", "(input_width * input_height)");
    ss << this->program_->define("TILE_K", this->kernel_shape_[1]);
    ss << this->program_->define("TILE_N", 32);
    ss << this->program_->define("OUT_PITCH_Z",
                               "(output_width * output_height * OUT_DEPTH)");
    ss << this->program_->define("ALIGNED_INPUT_SIZE",
                               "(input_height * input_width * INPUT_DEPTH)");

    vector<string> elems16({
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8", "s9", "sa", "sb", "sc", "sd", "se", "sf" });

#define GENERATE_DOT_PRODUCT(ele_num) \
    do { \
    ss << "#define DOT_PRODUCT_" << ele_num \
       << "( _result, _rowA, colB ) {   "; \
    for (int_tp i = 0; i < ele_num; i++) { \
      if (i < 10) {\
        ss << "_result.s" << i \
           << " = mad( _rowA, sub_group_broadcast( colB, " << i \
           << "), _result.s" << i << " );"; \
      } else {\
        ss << "_result." << elems16[i] \
           << " = mad( _rowA, sub_group_broadcast( colB, " << i \
           << "), _result." << elems16[i] << " );"; \
      }\
    } \
    ss << "    }" << std::endl; \
    } while (0)

    GENERATE_DOT_PRODUCT(8);
    GENERATE_DOT_PRODUCT(16);

    // kernel source
    if (simd_size == 8)
       ss << "__attribute__((intel_reqd_sub_group_size(8)))" << std::endl;
    else if (!IsBeignet(&ctx))
       ss << "__attribute__((intel_reqd_sub_group_size(16)))" << std::endl;
    ss << "__kernel void Conv_Interleaved(" << std::endl;
    ss << "const __global MItype *src0," << std::endl;
    ss << "const __global MItype *src1," << std::endl;
    ss << "const __global MItype *biases," << std::endl;
    ss << "__global MItype *dst," << std::endl;
    ss << "const ushort input_width," << std::endl;
    ss << "const ushort input_height," << std::endl;
    ss << "const ushort output_width," << std::endl;
    ss << "const ushort output_height)" << std::endl;
    ss << "{" << std::endl;
    ss << "const int_tp group_x = get_group_id(0);" << std::endl;
    ss << "const int_tp group_y = get_group_id(1);" << std::endl;
    ss << "const int_tp global_x = get_global_id(0);" << std::endl;
    ss << "const int_tp global_y = get_global_id(1);" << std::endl;
    ss << "const int_tp global_z = get_global_id(2);" << std::endl;
    ss << "int_tp interleaved_y;" << std::endl;
    ss << "int_tp kernel_y;" << std::endl;
    ss << "int_tp kernel_idx;" << std::endl;
    ss << "typedef CAT( MItype, KERNEL_WIDTH ) Dtype_t;" << std::endl;
    // True for all threads if filter_width is multiple of TILE_N
    // else, true for all but right-most column of threads.
    ss << "if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N ) " << std::endl;
    ss << "{" << std::endl;
    // Result ctile (*dst) is M rows X N columns
    // LWG size is 1x8 or 1x16.
    // Thus each thread calculates (8 or 16) *M rows X N cols of ctile.
    if (simd_size == 16) {
      ss << "Dtype16  blockC00 = 0.f;" << std::endl;
      ss << "Dtype16  blockC10 = 0.f;" << std::endl;
    } else {
      ss << "Dtype8  blockC00 = 0.f;" << std::endl;
      ss << "Dtype8  blockC10 = 0.f;" << std::endl;
      ss << "Dtype8  blockC20 = 0.f;" << std::endl;
      ss << "Dtype8  blockC30 = 0.f;" << std::endl;
    }
    if (blockM == 2 && simd_size == 8) {
      ss << "Dtype8  blockC01 = 0.f;" << std::endl;
      ss << "Dtype8  blockC11 = 0.f;" << std::endl;
      ss << "Dtype8  blockC21 = 0.f;" << std::endl;
      ss << "Dtype8  blockC31 = 0.f;" << std::endl;
    }
    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows X K columns." << std::endl
    ss << "int_tp curr_x = ( (global_y * TILE_M) % output_width ) * STRIDE_X;"
       << std::endl;
    ss << "int_tp curr_y = ( (global_y * TILE_M) / output_width ) * STRIDE_Y;"
       << std::endl;
    if (blockM == 2) {
      ss << "int_tp curr_x1 = ((global_y * TILE_M + 1)"
         << " % output_width) * STRIDE_X;"
         << std::endl;
      ss << "int_tp curr_y1 = ((global_y * TILE_M + 1)"
         << " / output_width) * STRIDE_Y;"
         << std::endl;
    }
    if (this->pad_[0] != 0 || this->pad_[1] != 0
        || this->dilation_[1] != 1 || this->dilation_[0] != 1) {
      ss << "int_tp saved_y = curr_y;" << std::endl;
      if (blockM == 2) {
        ss << "int_tp saved_y1 = curr_y1;" << std::endl;
      }
    }
    ss << "const __global MItype *src0_read = src0" << std::endl;
    // batch offset
    ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
    // Y offset
    ss << "+ (curr_y - INPUT_PAD_H) * ROW_PITCH" << std::endl;
    // X offset
    ss << "+ (curr_x - INPUT_PAD_W);" << std::endl;
    if (blockM == 2) {
      ss << "const __global MItype *src0_read1 = src0" << std::endl;
      // batch offset
      ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
      // Y offset
      ss << "+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH" << std::endl;
      // X offset
      ss << "+ curr_x1 - INPUT_PAD_W;" << std::endl;
    }
    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows X N columns.
    ss << "const __global MItype *src1_read = src1 + ( global_x * TILE_N  * 2);"
       << std::endl;
    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    ss << "int_tp patch_depth = 0;" << std::endl;
    if (!IsBeignet(&ctx) && simd_size == 16)
      ss << "__attribute__((opencl_unroll_hint(1)))" << std::endl;
    ss << "do" << std::endl;
    ss << "{" << std::endl;
    ss << "int_tp patch_row = 0;" << std::endl;
    if (this->pad_[0] != 0 || this->pad_[1] != 0
        || this->dilation_[1] != 1 || this->dilation_[0] != 1) {
      ss << "curr_y = saved_y;" << std::endl;
      if (blockM == 2)
        ss << "curr_y1 = saved_y1;" << std::endl;
    }
    if (!IsBeignet(&ctx) && simd_size == 16)
      ss << "__attribute__((opencl_unroll_hint(1)))" << std::endl;
    ss << "do" << std::endl;
    ss << "{" << std::endl;
    /*
     * Load atile and btile.
     *
     * Kernel data is partially interleaved. 
     * Every 2 rows are interleaved at Dtype8 granularity.
     * The exception is that if KERNEL_WIDTH is odd the last row is not
     * interleaved.
     * The non interleaved row is padded with zero to ensure same size
     * as interleaved rows.
     * This interleaving is done to ensure 0% GDR bank conflicts.
     * For example, this is how the
     * kernel data would be arranged before/after interleaving for
     * KERNEL_WIDTH=3.
     * (0, 0) (8, 0) (16, 0) (24, 0) ...    (0, 0) (0, 1) (8, 0) (8, 1)
     * (0, 1) (8, 1) (16, 1) (24, 1) ... => (0, 2) (8, 2) (16, 2) (24, 2) ...
     * (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
     * ...
     */
    ss << "const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;"
       << std::endl;
    if (this->pad_[0] == 0 && this->pad_[1] == 0
        && this->dilation_[1] == 1 && this->dilation_[0] == 1) {
      ss << "Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[0];"
         << std::endl;
      ss << "MItype*  pblockA00 = (MItype*)(&blockA00);" << std::endl;
      if (blockM == 2) {
        ss << "Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[0];"
           << std::endl;
        ss << "MItype*  pblockA01 = (MItype*)(&blockA01);" << std::endl;
      }
    } else {
      ss << "Dtype_t blockA00;" << std::endl;
      ss << "MItype*  pblockA00 = (MItype*)(&blockA00);" << std::endl;
      ss << "int_tp pos = 0;" << std::endl;
      ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
      ss << "{" << std::endl;
      ss << "if (curr_y >= INPUT_PAD_H && "
         << "curr_y < input_height + INPUT_PAD_H && "
         << "curr_x + pos * DILATION_X >= INPUT_PAD_W && "
         << "curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)"
         << std::endl;
      ss << "pblockA00[pos] = src0_read[pos * DILATION_X];" << std::endl;
      ss << "else" << std::endl;
      ss << "pblockA00[pos] = 0;" << std::endl;
      ss << "})" << std::endl;
      ss << "curr_y += DILATION_Y;" << std::endl;
      if (blockM == 2) {
        ss << "Dtype_t blockA01;" << std::endl;
        ss << "MItype*  pblockA01 = (MItype*)(&blockA01);" << std::endl;
        ss << "pos = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
        ss << "{" << std::endl;
        ss << "if (curr_y1 >= INPUT_PAD_H && "
           << "curr_y1 < input_height + INPUT_PAD_H && "
           << "curr_x1 + pos * DILATION_X >= INPUT_PAD_W && "
           << "curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)"
           << std::endl;
        ss << "pblockA01[pos] = src0_read1[pos * DILATION_X];" << std::endl;
        ss << "else" << std::endl;
        ss << "pblockA01[pos] = 0;" << std::endl;
        ss << "})" << std::endl;
        ss << "curr_y1 += DILATION_Y;" << std::endl;
      }
    }
    ss << "src0_read += (ROW_PITCH * DILATION_Y);" << std::endl;
    if (blockM == 2) {
      ss << "src0_read1 += (ROW_PITCH * DILATION_Y);" << std::endl;
    }
    ss << "uint blockB00[KERNEL_WIDTH * (TILE_N_PER_LANE)];" << std::endl;
    ss << "Dtype8* p8BlockB00 = (Dtype8*)blockB00;" << std::endl;
    ss << "Dtype4* p4BlockB00 = (Dtype4*)blockB00;" << std::endl;
    ss << "Dtype2* p2BlockB00 = (Dtype2*)blockB00;" << std::endl;
    ss << "MItype*  pBlockB00 =  (MItype* )blockB00;" << std::endl;
    ss << "interleaved_y = 0;" << std::endl;
    ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
    ss << "{ " << std::endl;
    if (simd_size == 8) {
      ss << "p8BlockB00[interleaved_y] = as_Dtype8("
         << "SUB_GROUP_BLOCK_READ8( (const __global INT_TYPE *)src1_read ) ); "
         << std::endl;
    } else {
      ss << "p4BlockB00[interleaved_y] = as_Dtype4("
         << "SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) ); "
         << std::endl;
    }
    ss << "src1_read += WIDTH1 * 2;" << std::endl;
    ss << "} )" << std::endl;
    ss << "if ( kernel_width_is_odd )" << std::endl;
    ss << "{" << std::endl;
    if (simd_size == 8) {
      ss << "p4BlockB00[KERNEL_WIDTH - 1] = as_Dtype4("
         << "SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) ); "
         << std::endl;
    } else {
      ss << "p2BlockB00[KERNEL_WIDTH - 1] = as_Dtype2("
         << "SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE *)src1_read ) ); "
         << std::endl;
    }
    ss << "src1_read += WIDTH1 * 2;" << std::endl;
    ss << "}" << std::endl;
    ss << "// Perform MADs" << std::endl;
    ss << "kernel_idx = 0;" << std::endl;
    ss << "interleaved_y = 0;" << std::endl;
    ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
    ss << "{" << std::endl;
    ss << "kernel_y = interleaved_y * 2;" << std::endl;
    if (simd_size == 16) {
      ss << "DOT_PRODUCT_16("
         << "blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); "
         << "kernel_idx++;"
         << std::endl;
      ss << "DOT_PRODUCT_16("
         << "blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); "
         << "kernel_idx++;"
         << std::endl;
      ss << "DOT_PRODUCT_16("
         << "blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); "
         << "kernel_idx++;"
         << std::endl;
      ss << "DOT_PRODUCT_16("
         << "blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); "
         << "kernel_idx++;"
         << std::endl;
    } else {
      ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
    }
    if (blockM == 2) {
      ss << "kernel_idx -= 8;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
    }
    ss << "} )" << std::endl;
    ss << "kernel_y = interleaved_y * 2;" << std::endl;
    ss << "if ( kernel_width_is_odd )" << std::endl;
    ss << "{" << std::endl;
    if (simd_size == 16) {
      ss << "DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
    } else {
      ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
    }
    if (blockM == 2) {
      ss << "kernel_idx -= 4;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], "
         << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "while( ++patch_row < KERNEL_HEIGHT );" << std::endl;
    // reset to start of next slice of patch
    ss << "src0_read += "
       << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);"
       << std::endl;
    if (blockM == 2) {
    // reset to start of next slice of patch
      ss << "src0_read1 += "
         << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);"
         << std::endl;
    }
    ss << "} " << std::endl;
    ss << "while ( ++patch_depth < INPUT_DEPTH );" << std::endl;
    // Dst resembles a cube of width X height X (output channel * batches).
    // Each tile writes: (SIMD * TILE_M) X 1 X TILE_N.
    // Partial writes most likely generated if padding used.
    ss << "__global MItype *out = dst " << std::endl;
    // batch offset
    ss << "+ global_z * OUT_PITCH_Z" << std::endl;
    // channel offset
    ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
    // Y offset
    ss << "+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * "
       << "OUT_PITCH_X" << std::endl;
    // X offset
    ss << "+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;"
       << std::endl;
    if (blockM == 2) {
      ss << "__global MItype *out1 = dst " << std::endl;
      ss << "+ global_z * OUT_PITCH_Z" << std::endl;
      ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
      ss << "+ ((global_y * TILE_M + 1) / output_width + OUT_PADDING_HEIGHT)*"
         << "OUT_PITCH_X" << std::endl;
      ss << "+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;"
         << std::endl;
    }
    ss << "MItype bias[TILE_N_PER_LANE];" << std::endl;
    ss << "typedef CAT( MItype, TILE_N_PER_LANE) Dtype_flex;" << std::endl;
    ss << "Dtype_flex *bias_vec;" << std::endl;
    ss << "bias_vec = (Dtype_flex*)bias;" << std::endl;
    if (simd_size == 16) {
      ss << "*bias_vec = "
         << "as_Dtype2(SUB_GROUP_BLOCK_READ42("
         << "(__global INT_TYPE *)biases + group_x * TILE_N));"
         << std::endl;
      // Work around a potential compiler bug
      ss << "if (group_x > 0xFFFFFFFEul)" << std::endl;
      ss << "out[0] = bias[0] + bias[1];" << std::endl;
    } else {
      ss << "*bias_vec = "
         << "as_Dtype4(SUB_GROUP_BLOCK_READ4("
         << "(__global INT_TYPE *)biases + group_x * TILE_N));"
         << std::endl;
    }
    ss << "if (global_y * TILE_M < output_width * output_height )" << std::endl;
    ss << "{" << std::endl;
    if (simd_size == 16) {
      ss << "for (int_tp i = 0; i < 16; i++)" << std::endl;
      ss << "{" << std::endl;
      ss << "out[( 0+i) * OUT_PITCH_Y] = "
         << "blockC00[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
      ss << "out[(16+i) * OUT_PITCH_Y] = "
         << "blockC10[i] + intel_sub_group_shuffle(bias[1], i);;" << std::endl;
      ss << "}" << std::endl;
    } else {
      ss << "for (int_tp i = 0; i < 8; i++)" << std::endl;
      ss << "{" << std::endl;
      ss << "out[( 0+i) * OUT_PITCH_Y] = "
         << "blockC00[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
      ss << "out[( 8+i) * OUT_PITCH_Y] = "
         << "blockC10[i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
      ss << "out[(16+i) * OUT_PITCH_Y] = "
         << "blockC20[i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
      ss << "out[(24+i) * OUT_PITCH_Y] = "
         << "blockC30[i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
      ss << "}" << std::endl;
    }
    if (blockM == 2) {
      ss << "if( global_y * TILE_M + 1 < output_width * output_height )"
         << std::endl;
      ss << "{" << std::endl;
      ss << "for( int_tp i = 0; i < 8; i++ )" << std::endl;
      ss << "{" << std::endl;
      ss << "out1[( 0+i) * OUT_PITCH_Y] = "
         << "blockC01[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
      ss << "out1[( 8+i) * OUT_PITCH_Y] = "
         << "blockC11[i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
      ss << "out1[(16+i) * OUT_PITCH_Y] = "
         << "blockC21[i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
      ss << "out1[(24+i) * OUT_PITCH_Y] = "
         << "blockC31[i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
      ss << "}" << std::endl;
      ss << "}" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "#if TILE_N_LAST > 0" << std::endl;
    ss << "else" << std::endl;
    ss << "{" << std::endl;
    // Result ctile (*dst) is M rows X N columns
    // LWG size is 1x8.  Thus each thread calculates 8*M rows X N cols of ctile.
    ss << "int_tp i = 0;" << std::endl;
    ss << "Dtype8  blockC[TILE_N_LAST_DIV8];" << std::endl;
    ss << "LOOP(TILE_N_LAST_DIV8, i," << std::endl;
    ss << "{" << std::endl;
    ss << "blockC[i] = 0.f;" << std::endl;
    ss << "} )" << std::endl;
    ss << "int_tp curr_x = ( global_y % output_width ) * STRIDE_X;"
       << std::endl;
    ss << "int_tp curr_y = ( global_y / output_width ) * STRIDE_Y;"
       << std::endl;
    if (this->pad_[0] != 0 || this->pad_[1] != 0
        || this->dilation_[1] != 1 || this->dilation_[0] != 1) {
      ss << "int_tp saved_y = curr_y;" << std::endl;
    }
    ss << "const __global MItype *src0_read = src0" << std::endl;
    ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
    ss << "+ (curr_y - INPUT_PAD_H) * ROW_PITCH" << std::endl;
    ss << "+ (curr_x - INPUT_PAD_W);" << std::endl;
    if (blockM == 2) {
      ss << "i = 0;" << std::endl;
      ss << "Dtype8  blockC1[TILE_N_LAST_DIV8];" << std::endl;
      ss << "LOOP(TILE_N_LAST_DIV8, i," << std::endl;
      ss << "{" << std::endl;
      ss << "blockC1[i] = 0.f;" << std::endl;
      ss << "} )" << std::endl;
      ss << "int_tp curr_x1 = ((global_y * TILE_M + 1)"
         << " % output_width) * STRIDE_X;"
         << std::endl;
      ss << "int_tp curr_y1 = ((global_y * TILE_M + 1)"
         << " / output_width) * STRIDE_Y;"
         << std::endl;
      if (this->pad_[0] != 0 || this->pad_[1] != 0
          || this->dilation_[1] != 1 || this->dilation_[0] != 1) {
        ss << "int_tp saved_y1 = curr_y1;" << std::endl;
      }
      ss << "const __global MItype *src0_read1 = src0" << std::endl;
      ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
      ss << "+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH" << std::endl;
      ss << "+ (curr_x1 - INPUT_PAD_W);" << std::endl;
    }
    ss << "const __global MItype *src1_read = src1 + ( global_x * TILE_N  * 2);"
       << std::endl;
    ss << "int_tp patch_depth = 0;" << std::endl;
    ss << "do" << std::endl;
    ss << "{" << std::endl;
    ss << "int_tp patch_row = 0;" << std::endl;
    if (this->pad_[0] != 0 || this->pad_[1] != 0
        || this->dilation_[1] != 1 || this->dilation_[0] != 1) {
      ss << "curr_y = saved_y;" << std::endl;
      if (blockM == 2) {
        ss << "curr_y1 = saved_y1;" << std::endl;
      }
    }
    ss << "do" << std::endl;
    ss << "{" << std::endl;
    ss << "const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;"
       << std::endl;
    if (this->pad_[0] == 0 && this->pad_[1] == 0
        && this->dilation_[1] == 1 && this->dilation_[0] == 1) {
      ss << "Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[0];"
         << std::endl;
      ss << "MItype*  pblockA00 = (MItype*)(&blockA00);" << std::endl;
      if (blockM == 2) {
        ss << "Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[0];"
           << std::endl;
        ss << "MItype*  pblockA01 = (MItype*)(&blockA01);" << std::endl;
      }
    } else {
      ss << "Dtype_t blockA00;" << std::endl;
      ss << "MItype*  pblockA00 = (MItype*)(&blockA00);" << std::endl;
      ss << "int_tp pos = 0;" << std::endl;
      ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
      ss << "{" << std::endl;
      ss << "if (curr_y >= INPUT_PAD_H && "
         << "curr_y < input_height + INPUT_PAD_H && "
         << "curr_x + pos * DILATION_X >= INPUT_PAD_W && "
         << "curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)"
         << std::endl;
      ss << "pblockA00[pos] = src0_read[pos * DILATION_X];" << std::endl;
      ss << "else" << std::endl;
      ss << "pblockA00[pos] = 0;" << std::endl;
      ss << "})" << std::endl;
      ss << "curr_y += DILATION_Y;" << std::endl;
      if (blockM == 2) {
        ss << "Dtype_t blockA01;" << std::endl;
        ss << "MItype*  pblockA01 = (MItype*)(&blockA01);" << std::endl;
        ss << "pos = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
        ss << "{" << std::endl;
        ss << "if (curr_y1 >= INPUT_PAD_H && "
           << "curr_y1 < input_height + INPUT_PAD_H && "
           << "curr_x1 + pos * DILATION_X >= INPUT_PAD_W && "
           << "curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)"
           << std::endl;
        ss << "pblockA01[pos] = src0_read1[pos * DILATION_X];" << std::endl;
        ss << "else" << std::endl;
        ss << "pblockA01[pos] = 0;" << std::endl;
        ss << "})" << std::endl;
        ss << "curr_y1 += DILATION_Y;" << std::endl;
      }
    }
    ss << "src0_read += (ROW_PITCH * DILATION_Y);" << std::endl;
    if (blockM == 2) {
      ss << "src0_read1 += (ROW_PITCH * DILATION_Y);" << std::endl;
    }
    ss << "MItype blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];" << std::endl;
    ss << "interleaved_y = 0;" << std::endl;
    ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
    ss << "{ " << std::endl;
    ss << "#if TILE_N_LAST_DIV8 == 1" << std::endl;
    ss << "Dtype2* p2BlockB = (Dtype2* )blockB;" << std::endl;
    ss << "p2BlockB[interleaved_y] = as_Dtype2("
       << "SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "#elif TILE_N_LAST_DIV8 == 2" << std::endl;
    ss << "Dtype4* p4BlockB = (Dtype4* )blockB;" << std::endl;
    ss << "p4BlockB[interleaved_y] = as_Dtype4("
       << "SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "#elif TILE_N_LAST_DIV8 == 3" << std::endl;
    ss << "//TODO: broken.  No block_read6" << std::endl;
    ss << "Dtype6* p6BlockB = (Dtype6* )blockB;" << std::endl;
    ss << "(*((Dtype8*)(&p6BlockB[interleaved_y]))).s0123 = as_Dtype4("
       << "SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "(*((Dtype8*)(&p6BlockB[interleaved_y]))).s45 = as_Dtype2("
       << "SUB_GROUP_BLOCK_READ2("
       << "(const __global INT_TYPE *)(src1_read + 4 * 8)));" << std::endl;
    ss << "#endif" << std::endl;
    ss << "src1_read += WIDTH1 * 2;" << std::endl;
    ss << "} )" << std::endl;
    ss << "if ( kernel_width_is_odd )" << std::endl;
    ss << "{" << std::endl;
    ss << "#if TILE_N_LAST_DIV8 == 1" << std::endl;
    ss << "MItype* pBlockB = (MItype* )blockB;" << std::endl;
    ss << "pBlockB[KERNEL_WIDTH - 1] = as_Dtype("
       << "SUB_GROUP_BLOCK_READ( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "#elif TILE_N_LAST_DIV8 == 2" << std::endl;
    ss << "Dtype2* p2BlockB = (Dtype2* )blockB;" << std::endl;
    ss << "p2BlockB[KERNEL_WIDTH - 1] = as_Dtype2("
       << "SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "#elif TILE_N_LAST_DIV8 == 3" << std::endl;
    ss << "Dtype3* p3BlockB = (Dtype3* )blockB;" << std::endl;
    ss << "p3BlockB[KERNEL_WIDTH - 1].s01 = as_Dtype2("
       << "SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE *)src1_read ) );"
       << std::endl;
    ss << "p3BlockB[KERNEL_WIDTH - 1].s2 = as_Dtype("
       << "SUB_GROUP_BLOCK_READ( (const __global INT_TYPE *)"
       << "(src1_read + 2 * 8)));" << std::endl;
    ss << "#endif" << std::endl;
    ss << "src1_read += WIDTH1 * 2;" << std::endl;
    ss << "}" << std::endl;
    ss << "// Perform MADs" << std::endl;
    ss << "MItype* pBlockB = (MItype*)blockB;" << std::endl;
    ss << "kernel_idx = 0;" << std::endl;
    ss << "interleaved_y = 0;" << std::endl;
    ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
    ss << "{" << std::endl;
    ss << "kernel_y = interleaved_y * 2;" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y    ],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y + 1],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y    ],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y + 1],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y    ],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y + 1],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#endif" << std::endl;
    if (blockM == 2) {
      ss << "kernel_idx -= TILE_N_LAST_DIV8 * 2;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y    ],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y + 1],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y    ],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y + 1],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y    ],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y + 1],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#endif" << std::endl;
      ss << "#endif" << std::endl;
    }
    ss << "} )" << std::endl;
    ss << "kernel_y = interleaved_y * 2;" << std::endl;
    ss << "if ( kernel_width_is_odd )" << std::endl;
    ss << "{" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
    ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y],"
       << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#endif" << std::endl;
    if (blockM == 2) {
      ss << "kernel_idx -= TILE_N_LAST_DIV8;" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
      ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y],"
         << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
      ss << "#endif" << std::endl;
      ss << "#endif" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "//while( ++patch_row < 1 ); //debug" << std::endl;
    ss << "while( ++patch_row < KERNEL_HEIGHT );" << std::endl;
    ss << "src0_read += "
       << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );"
       << std::endl;
    ss << "} " << std::endl;
    ss << "while ( ++patch_depth < INPUT_DEPTH );" << std::endl;
    ss << "__global MItype *out = dst " << std::endl;
    ss << "+ global_z * OUT_PITCH_Z" << std::endl;
    ss << "+ (group_x * TILE_N) * OUT_PITCH_Y" << std::endl;
    ss << "+ ((global_y * TILE_M) / output_width + "
       << "OUT_PADDING_HEIGHT) * OUT_PITCH_X" << std::endl;
    ss << "+ ((global_y * TILE_M) % output_width ) + OUT_PADDING_LEFT;"
       << std::endl;
    if (blockM == 2) {
      ss << "__global MItype *out1 = dst " << std::endl;
      ss << "+ global_z * OUT_PITCH_Z" << std::endl;
      ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
      ss << "+ ((global_y * TILE_M + 1) / output_width + OUT_PADDING_HEIGHT ) *"
         << "OUT_PITCH_X" << std::endl;
      ss << "+ ((global_y * TILE_M + 1) % output_width ) + OUT_PADDING_LEFT;"
         << std::endl;
    }
    ss << "MItype bias[4];" << std::endl;
    ss << "Dtype4 *bias_vec;" << std::endl;
    ss << "bias_vec = (Dtype4*)bias;" << std::endl;
    ss << "*bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4("
       << "(__global INT_TYPE *)biases + group_x * TILE_N));" << std::endl;
    ss << "if (global_y * TILE_M < output_width * output_height )" << std::endl;
    ss << "{" << std::endl;
    ss << "for (int_tp i = 0; i < 8; i++)" << std::endl;
    ss << "{" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 0 ) out[( 0+i) * OUT_PITCH_Y] = "
       << "blockC[0][i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 1 ) out[( 8+i) * OUT_PITCH_Y] = "
       << "blockC[1][i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 2 ) out[(16+i) * OUT_PITCH_Y] = "
       << "blockC[2][i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 3 ) out[(24+i) * OUT_PITCH_Y] = "
       << "blockC[3][i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    if (blockM == 2) {
    ss << "if( global_y * TILE_M + 1 < output_width * output_height )"
       << std::endl;
    ss << "{" << std::endl;
    ss << "for( int_tp i = 0; i < 8; i++ )" << std::endl;
    ss << "{" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 0 ) out1[( 0+i) * OUT_PITCH_Y] = "
       << "blockC1[0][i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 1 ) out1[( 8+i) * OUT_PITCH_Y] = "
       << "blockC1[1][i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 2 ) out1[(16+i) * OUT_PITCH_Y] = "
       << "blockC1[2][i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
    ss << "if ( TILE_N_LAST_DIV8 > 3 ) out1[(24+i) * OUT_PITCH_Y] = "
       << "blockC1[3][i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "#endif" << std::endl;
    ss << "}" << std::endl;
  } else if (kernelType == KERNEL_TYPE_BASIC) {
    kernelUKey = generate_specific_key(4, blockM, blockK, blockN);
    kernel_name_ = "BASIC_";
    kernel_name_ += kernelUKey.c_str();

    // opts
    opts << " -cl-fast-relaxed-math -D CFMultiNoPadding=" << kernel_name_;
    if (IsBeignet(&ctx))
      opts << " -D__BEIGNET__ ";
    options_ = opts.str();

    // defs
    ss << this->program_->define("CHANNELS", this->fmaps_in_ / this->group_);
    ss << this->program_->define("APPLY_BIAS", this->bias_term_);
    ss << this->program_->define("OUTPUT_Z", this->M_FW_);
    ss << this->program_->define("ZPAR", 1);

    // kernel
    ss << "#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) "
       << "do { (_dst_)[(_offset_)] = (_data_);} while(0)" << std::endl;
    ss << "__kernel void CFMultiNoPadding(" << std::endl;
    ss << "__global MItype* image_data," << std::endl;
    ss << "int_tp image_offset," << std::endl;
    ss << "__global MItype* kernel_data, " << std::endl;
    ss << "int_tp kernel_offset," << std::endl;
    ss << "__global MItype* bias," << std::endl;
    ss << "const int_tp bias_offset," << std::endl;
    ss << "__global MItype* convolved_image, " << std::endl;
    ss << "const int_tp convolved_image_offset," << std::endl;
    ss << "const ushort input_width," << std::endl;
    ss << "const ushort input_height," << std::endl;
    ss << "const ushort output_width," << std::endl;
    ss << "const ushort output_height," << std::endl;
    ss << "const ushort pad_w," << std::endl;
    ss << "const ushort pad_h) {" << std::endl;
    ss << "const int_tp outputX = get_global_id(0);" << std::endl;
    ss << "const int_tp outputY = get_global_id(1);" << std::endl;
    ss << "const int_tp kernelNum = get_global_id(2)*ZPAR;" << std::endl;
    ss << "if(outputX < output_width && outputY < output_height)" << std::endl;
    ss << "{" << std::endl;
    ss << "MItype sum[ZPAR];" << std::endl;
    ss << "for(int_tp kern =0; kern < ZPAR; kern++)" << std::endl;
    ss << "{" << std::endl;
    ss << "sum[kern] = 0.0f;" << std::endl;
    ss << "}" << std::endl;
    ss << "const int_tp org_y = outputY * STRIDE_Y - pad_h;" << std::endl;
    ss << "const int_tp org_x = outputX * STRIDE_X - pad_w;" << std::endl;
    ss << "const int_tp currentKernelOffset = "
       << "kernel_offset + kernelNum*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS;"
       << std::endl;
    ss << "const int_tp biasIndex=bias_offset + kernelNum;" << std::endl;
    ss << "const int_tp local_image_offset = org_y*input_width + org_x;"
       << std::endl;
    ss << "const int_tp imageSize = input_width*input_height;" << std::endl;
    ss << "__global MItype* image_dataPtrFloat = "
       << "(image_data + (image_offset + local_image_offset));" << std::endl;
    ss << "__global MItype* kernel_dataPtrFloat = "
       << "(kernel_data + (currentKernelOffset));" << std::endl;
    ss << "for(int_tp c = 0; c < CHANNELS; c++)" << std::endl;
    ss << "{" << std::endl;
    ss << "for(int_tp Y = 0; Y < KERNEL_HEIGHT; Y++)" << std::endl;
    ss << "{" << std::endl;
    ss << "for(int_tp X = 0; X < KERNEL_WIDTH; X++)" << std::endl;
    ss << "{" << std::endl;
    ss << "if(!(org_y + Y * DILATION_Y >= 0 && "
       << "org_y + Y * DILATION_Y < input_height && "
       << "org_x + X * DILATION_X >= 0 && "
       << "org_x + X * DILATION_X < input_width))" << std::endl;
    ss << "{" << std::endl;
    ss << "continue;" << std::endl;
    ss << "}" << std::endl;
    ss << "for(int_tp kern =0; kern < ZPAR; kern++)" << std::endl;
    ss << "{" << std::endl;
    ss << "sum[kern] += image_dataPtrFloat[X * DILATION_X] * "
       << "kernel_dataPtrFloat[kern*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS + X];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "image_dataPtrFloat += input_width * DILATION_Y;" << std::endl;
    ss << "kernel_dataPtrFloat += KERNEL_WIDTH;" << std::endl;
    ss << "}" << std::endl;
    ss << "image_dataPtrFloat += "
       << "imageSize - input_width*KERNEL_HEIGHT*DILATION_Y;" << std::endl;
    ss << "}" << std::endl;
    ss << "if(APPLY_BIAS == 1)" << std::endl;
    ss << "{" << std::endl;
    ss << "for(int_tp kern = 0; kern < ZPAR; kern++)" << std::endl;
    ss << "{" << std::endl;
    ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
    ss << "{" << std::endl;
    ss << "int_tp offset = convolved_image_offset + "
       << "(kernelNum+kern)*output_height*output_width + "
       << "outputY*output_width + outputX;" << std::endl;
    ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern] + "
       << "bias[biasIndex +kern]);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "else" << std::endl;
    ss << "{" << std::endl;
    ss << "for(int_tp kern = 0; kern < ZPAR; kern++)" << std::endl;
    ss << "{" << std::endl;
    ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
    ss << "{" << std::endl;
    ss << "int_tp offset = convolved_image_offset + "
       << "(kernelNum+kern)*output_height*output_width + "
       << "outputY*output_width + outputX;" << std::endl;
    ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::GenerateHelperKernels() {
  stringstream ss;

  ss << LibDNN<MItype>::generate_header();
  ss << "#define CONCAT(a,b) a##_##b" << std::endl;
  ss << "#define TEMPLATE(name,type) CONCAT(name,type)" << std::endl;
  ss << "__kernel void TEMPLATE(copyWeightsSwizzled, MItype)" << std::endl;
  ss << "(__global MItype* weightIn," << std::endl;
  ss << "__global MItype* weightOut," << std::endl;
  ss << "const int_tp kernel_w," << std::endl;
  ss << "const int_tp kernel_h," << std::endl;
  ss << "const int_tp channels," << std::endl;
  ss << "const int_tp outputs," << std::endl;
  ss << "const int_tp swizzleFactor) {" << std::endl;
  ss << "uint_tp sX = get_global_id(0);" << std::endl;
  ss << "//Original location" << std::endl;
  ss << "//Output location" << std::endl;
  ss << "int_tp outputSublayer = channels / swizzleFactor;" << std::endl;
  ss << "int_tp outputSublayerIndex = channels % swizzleFactor;" << std::endl;
  ss << "int_tp filter = sX / (kernel_w*kernel_h*channels);" << std::endl;
  ss << "int_tp kernel_X = sX % kernel_w;" << std::endl;
  ss << "int_tp kernel_Y = (sX / kernel_w) % kernel_h;" << std::endl;
  ss << "int_tp kernel_C = (sX / (kernel_w * kernel_h)) % channels;"
     << std::endl;
  ss << "int_tp FP = filter / swizzleFactor;" << std::endl;
  ss << "int_tp F1 = filter % swizzleFactor;" << std::endl;
  ss << "weightOut[FP*(kernel_w*kernel_h*channels*swizzleFactor) + "
     << "kernel_C*(kernel_w*kernel_h*swizzleFactor) + "
     << "kernel_Y*(kernel_w*swizzleFactor) + "
     << "kernel_X*swizzleFactor + F1]" << std::endl;
  ss << "= weightIn[filter*(kernel_w*kernel_h*channels) + "
     << "kernel_C*(kernel_w*kernel_h) + "
     << "kernel_Y*kernel_w + kernel_X];" << std::endl;
  ss << "}" << std::endl;

  LibDNN<MItype>::kernel_ = ss.str();
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::GenerateKernels() {
  stringstream ss;

  ss << LibDNN<MItype>::generate_header();
  ss << generate_fw_defs();
  ss << generate_fw_kernels(kernelType_, blockM_, blockK_, blockN_);
  LibDNN<MItype>::kernel_ = ss.str();
}

template<typename MItype>
string LibDNNConvSpatial<MItype>string_identifier() {
  return NULL;
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::Forward(const MItype* bottom_data,
                                       const MItype* weight,
                                       const MItype* bias, MItype* top_data,
                                       int_tp batch_size) {
  weight_ = weight;
  if (this->bias_term_)
    bias_ = bias;
  bottom_data_ = bottom_data;
  top_data_ = top_data;
  bias_offset_ = 0;
  num_ = batch_size;

  if (!try_cache_) {
    load_cached_kernels(bottom_data, top_data);
    try_cache_ = true;
  }

  if (!tuned_)
    Tune(top_data, NULL, weight, NULL, bias, NULL,
         bottom_data, NULL, batch_size);

  convolve(bottom_data, top_data, 0, num_, bestKernelConfig);
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::Backward(bool prop_down_data,
                                        bool prop_down_weights,
                                        const MItype* top_data,
                                        const MItype* top_diff,
                                        const MItype* weight,
                                        MItype* weight_diff,
                                        const MItype* bias,
                                        MItype* bias_diff,
                                        const MItype* bottom_data,
                                        MItype* bottom_diff,
                                        int_tp batch_size) {
  this->libdnn_conv_.get()->Backward(prop_down_data,
                               prop_down_weights,
                               top_data,
                               top_diff,
                               weight,
                               weight_diff,
                               bias,
                               bias_diff,
                               bottom_data,
                               bottom_diff,
                               batch_size);
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::Tune(MItype* top_data, MItype* top_diff,
                                    const MItype* weight,
                                    MItype* weight_diff,
                                    const MItype* bias,
                                    MItype* bias_diff,
                                    const MItype* bottom_data,
                                    MItype* bottom_diff,
                                    int_tp batch_size) {
  cl_int err;
  MItype *verify_data;
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());

  verify_data = reinterpret_cast<MItype*>(clCreateBuffer(ctx.handle().get(),
           CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
           batch_size * this->fmaps_out_ * out_spatial_dim_ * sizeof(MItype),
           NULL, &err));
  CHECK_EQ(err, CL_SUCCESS) << "Failed to create verify buffer." << std::endl;

  calculate_verify_data(bottom_data, weight, bias, verify_data);
  setup_convolution(bottom_data, top_data, verify_data);
  clReleaseMemObject((cl_mem)verify_data);
  CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::calculate_verify_data(const MItype* bottom,
                                 const MItype* w,
                                 const MItype* bias,
                                 MItype* verify_data) {
  create_basic_kernel(bottom, verify_data, 1, 1, 1);
  kernel_index_ = kernelQueue.size() - 1;
  convolve(bottom, verify_data, 0, num_, kernelQueue[kernel_index_]);
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  clEnqueueCopyBuffer(ctx.get_queue().handle().get(),
                      (cl_mem)top_data_,
                      (cl_mem)verify_data, 0, 0,
                      sizeof(MItype) * num_ * this->top_dim_, 0, NULL, NULL);
  ctx.delete_program(kernelQueue[kernel_index_]->kernelName);
  kernelQueue.pop_back();
  return;
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::ForwardBenchmark(const MItype* bottom,
                                 const MItype* w,
                                 const MItype* bias,
                                 MItype* top,
                                 int_tp batch_size) {
  weight_ = w;
  if (this->bias_term_)
     bias_ = bias;
  bottom_data_ = bottom;
  top_data_ = top;
  bias_offset_ = 0;
  num_ = batch_size;
  calculate_verify_data(bottom, w, bias, top);
}

#define dbg
#ifdef dbg
#define dbgPrint(X) (X)
#else
#define dbgPrint(X)
#endif

// For large enough input size, we do not need to tune kernels for different
// size. The reason is with large input size, there will be enough work items
// to feed al the EUs.
// FIXME for the gemm like convolution, switch back to eaxct image size.

#define TUNING_SIZE(X) ((X) > 256 ? 256 : (ALIGN(X, 16)))

template<typename MItype>
void LibDNNConvSpatial<MItype>::generate_key() {
  CHECK((!std::is_same<MItype, double>::value));
  stringstream keyBuilder;
  if (std::is_same<MItype, float>::value)
    keyBuilder << "float_";
  else
    keyBuilder << "half_";
  // FIXME: to support fuse?
  keyBuilder << this->kernel_shape_[1] << "_"
             << this->kernel_shape_[0] << "_"
             << this->fmaps_in_ << "_"
             << this->group_ << "_"
             << this->stride_[0] << "_"
             << this->stride_[1] << "_"
             << this->dilation_[0] << "_"
             << this->dilation_[1] << "_"
             << this->bias_term_ << "_"
             << TUNING_SIZE(width_) << "_"
             << TUNING_SIZE(height_) << "_"
             << this->pad_[1] << "_"
             << this->pad_[0] << "_"
             << num_ << "_"
             << this->M_FW_;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                (LibDNN<MItype>::dev_ptr_->id());
  string prefix = ctx.current_device().name()
                  + ctx.current_device().vendor()
                  + ctx.current_device().driver_version()
                  + std::to_string(ctx.current_device().max_compute_units());
  key_ = viennacl::tools::sha1(prefix + keyBuilder.str());
  short_key_ = keyBuilder.str();
}

template<typename MItype>
string LibDNNConvSpatial<MItype>::generate_specific_key(
    int_tp type, int_tp blockWidth, int_tp blockHeight, int_tp blockDepth) {
  stringstream keyBuilder;
  CHECK((!std::is_same<MItype, double>::value));
  keyBuilder << short_key_
             << "_" << type
             << "_" << blockWidth
             << "_" << blockHeight
             << "_" << blockDepth;
  return keyBuilder.str();
}

template<typename MItype>
void interleaveMatrix(
         MItype* mem_dst, const MItype *mem,
         int_tp r, int_tp c, int_tp interleavedRows, int_tp nonInterleavedRows,
         int_tp blockWidth, int_tp rowAlignment ) {
  CHECK_EQ(interleavedRows % 2, 0) <<
      "interleaveMatrix only supports even values for interleavedRows.";

  size_t memSize = r * c * sizeof(MItype);
  size_t dstSize = memSize *
            (interleavedRows + nonInterleavedRows * 2) /
            (interleavedRows + nonInterleavedRows);
  memset(mem_dst, 0, dstSize);    // NOLINT

  const int_tp xStride = blockWidth;
  const int_tp yStride = c * 2;
  const MItype *pSrc = mem;
  MItype* pDst = mem_dst;
  for (int_tp Y = 0; Y < r;) {
    for (int_tp rows = 0; rows < interleavedRows; rows += 2) {
      if ( Y >= r ) break;
      if ((c % xStride) == 0) {
        for (int_tp X = 0; X < c / xStride; X++) {
          memcpy( pDst + X * xStride * 2,                         // NOLINT
                  pSrc + X * xStride,     xStride * sizeof(MItype));
          memcpy( pDst + X * xStride * 2 + xStride,               // NOLINT
                  pSrc + X * xStride + c, xStride * sizeof(MItype));
        }
      } else {
        const int_tp count = c / xStride;
        int_tp X = 0;
        for (; X < count - 1; X++) {
          memcpy(pDst + X * xStride * 2,                          // NOLINT
                 pSrc + X * xStride, xStride * sizeof(MItype));
          memcpy(pDst + X * xStride * 2 + xStride,                // NOLINT
                 pSrc + X * xStride + c, xStride * sizeof(MItype));
        }
        memcpy(pDst + X * xStride * 2,                            // NOLINT
               pSrc + X * xStride, xStride * sizeof(MItype));
      }
      pSrc += yStride;
      pDst += yStride;
      Y += 2;
    }

    for (int_tp rows = 0; rows < nonInterleavedRows; rows++) {
      if (Y >= r) break;
      const int_tp stride = rowAlignment;
      int_tp remaining = c;
      for (int_tp X = 0; X < c; X += stride) {
        if (remaining >= stride) {
          memcpy( pDst + X * 2, pSrc + X, stride * sizeof(MItype));    // NOLINT
          remaining -=stride;
        } else {
          memcpy(pDst + X * 2, pSrc + X, remaining * sizeof(MItype));  // NOLINT
        }
      }
      pSrc += yStride / 2;
      pDst += yStride;
      Y++;
    }
  }
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::swizzleWeights(
    const MItype *bottom,
    const MItype *top,
    int_tp swizzled_factor,
    bool interleave) {

  // Simply skip the weight swizzle if we already got a swizzled_weights_
  // in test phase and not in auto tuning
  // This requires we always call convolve again with the winner configuration
  // during the auto tuning stage.
  bool phase_test = this->get_config().phase_test;
  if (tuned_ &&
      swizzled_weights_ != NULL &&
      phase_test == true)
    return;

  cl_int err;
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  swizzled_weights_ = reinterpret_cast<MItype*>(
                      clCreateBuffer(ctx.handle().get(),
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                      sizeof(MItype) *
                      ((this->fmaps_out_ + 15) & ~15) *
                      this->fmaps_in_
                    * this->kernel_shape_[0]
                    * ((this->kernel_shape_[1] + 1) & ~1),
                      NULL, &err));
  CHECK_EQ(err, CL_SUCCESS) << "Failed to create swizzled_weights buffer.";

  if (!interleave) {
    viennacl::ocl::kernel &oclk_copy_weight =
       LibDNN<MItype>::ocl_program_.get_kernel(
             CL_KERNEL_SELECT("copyWeightsSwizzled"));
    cl_uint argIdx = 0;

    int_tp channels = this->fmaps_in_ / this->group_;
    oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) weight_, &ctx));
    oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) swizzled_weights_,
                         &ctx));
    oclk_copy_weight.arg(argIdx++, this->kernel_shape_[1]);
    oclk_copy_weight.arg(argIdx++, this->kernel_shape_[0]);
    oclk_copy_weight.arg(argIdx++, channels);
    oclk_copy_weight.arg(argIdx++, this->fmaps_out_);
    oclk_copy_weight.arg(argIdx++, swizzled_factor);
    const size_t global_work_size_Copy[3] = {
        (size_t) (ALIGN(this->fmaps_out_, swizzled_factor)
        * channels * this->kernel_shape_[1] * this->kernel_shape_[0]), 1, 1 };

    OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     oclk_copy_weight.handle().get(), 3, NULL,
                                     global_work_size_Copy, NULL, 0, NULL,
                                     NULL));
  } else {
    MItype* cpu_weight = reinterpret_cast<MItype*>(clEnqueueMapBuffer(
             ctx.get_queue().handle().get(), (cl_mem)weight_, true, CL_MAP_READ,
             0, sizeof(MItype) * this->fmaps_out_ * kernel_dim_ * this->group_,
             0, NULL, NULL, NULL));

    // assumption: kernel dimesion is 2
    MItype* cpu_swizzled_weight = reinterpret_cast<MItype*>(clEnqueueMapBuffer(
      ctx.get_queue().handle().get(),
      (cl_mem)swizzled_weights_,
      true, CL_MAP_WRITE, 0,
      sizeof(MItype) *
      ((this->fmaps_out_ + 15) & ~15) *
      this->fmaps_in_ * this->kernel_shape_[0]
                      * ((this->kernel_shape_[1] + 1) & ~1),
      0, NULL, NULL, NULL));

    int_tp interleavedRows = (this->kernel_shape_[1] / 2) * 2;
    int_tp nonInterleavedRows = this->kernel_shape_[1] % 2;
    int_tp blockWidth = swizzled_factor;  // should equal to simd size.
    int_tp rowAlignment = 32;
    size_t interleaved_filter_size =
       this->M_FW_ * this->kernel_shape_[1]
                   * this->kernel_shape_[0]
                   * this->fmaps_in_ * sizeof(MItype);
    MItype * tmpSwizzledWeight =
       reinterpret_cast<MItype*>(malloc(interleaved_filter_size));
    CHECK_EQ(tmpSwizzledWeight != NULL, true)
      << "Failed to allocate temporary swizzled weight";
    for (int_tp od = 0; od < this->M_FW_; od++)
      for (int_tp id = 0; id < this->fmaps_in_; id++)
        for (int_tp r = 0; r < this->kernel_shape_[0]; r++)
          for (int_tp c = 0; c < this->kernel_shape_[1]; c++)
            tmpSwizzledWeight[((id * this->kernel_shape_[0] + r)
                * this->kernel_shape_[1] + c) * this->M_FW_ + od]
            = cpu_weight[((od * this->fmaps_in_ + id)
            * this->kernel_shape_[0] + r)*this->kernel_shape_[1]+c];
    interleaveMatrix(cpu_swizzled_weight,
                     tmpSwizzledWeight,
                     this->kernel_shape_[1]
                   * this->kernel_shape_[0]
                   * this->fmaps_in_, this->M_FW_,
                     interleavedRows,
                     nonInterleavedRows,
                     blockWidth,
                     rowAlignment);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
                            (cl_mem)weight_,
                            cpu_weight, 0, NULL,
                            NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
                            (cl_mem)swizzled_weights_,
                            cpu_swizzled_weight, 0, NULL,
                            NULL);
    free(tmpSwizzledWeight);
  }
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::calculate_global_size(int_tp batch,
                                  int_tp* wio,  // work item output size
                                  size_t* lSize,  // local size
                                  size_t* gSize) {  // global size
  CHECK((!std::is_same<MItype, double>::value));
  gSize[0] = ceil(
      (fmax(static_cast<float>(output_w_) / wio[0], 1.0)) / lSize[0])
      * lSize[0];
  gSize[1] = ceil(
      (fmax(static_cast<float>(output_h_) / wio[1], 1.0)) / lSize[1])
      * lSize[1];
  gSize[2] = ceil(
      static_cast<float>(
          (ceil(static_cast<float>(this->M_FW_) * batch / wio[2])))
          / lSize[2]) * lSize[2];
}

template<typename MItype>
bool LibDNNConvSpatial<MItype>::create_basic_kernel(
    const MItype *bottom, const MItype *top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  int_tp workItemOutput[3];
  workItemOutput[0] = 1;
  workItemOutput[1] = 1;
  workItemOutput[2] = 1;

  kernelType_ = 4;
  blockM_ = blockWidth;
  blockK_ = blockHeight;
  blockN_ = blockDepth;
  GenerateKernels();
  compile_fw_kernel();

  size_t localSize[3] = { 1, 1, 1 };
  size_t globalSize[3];

  calculate_global_size(1, workItemOutput, localSize, globalSize);
  kernelQueue.push_back(
      new kernelConfig(kernel_name_, globalSize, localSize, workItemOutput,
                       false, false, true, 4));

  return true;
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::setBufferKernelArg(
    const MItype *bottom, const MItype *top,
    viennacl::ocl::kernel *kernel,
    const cl_uint &argIdx,
    viennacl::ocl::context *ctx,
    cl_mem buffer, size_t offset,
    size_t size, bool readOnly,
    bool preserved) {

  CHECK((!std::is_same<MItype, double>::value));
  if (offset == 0) {
    kernel->arg(argIdx, WrapHandle((cl_mem) buffer, ctx));
    return;
  }

  if (preserved &&
    subBufferMap.find(std::make_tuple(buffer, offset, size))
      != subBufferMap.end()) {
    kernel->arg(argIdx,
      WrapHandle(subBufferMap.find
                   (std::make_tuple(buffer, offset, size))->second, ctx));
    return;
  }
  cl_buffer_region region;
  region.origin = offset * sizeof(MItype);
  region.size = size * sizeof(MItype);
  cl_mem_flags memFlags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  cl_int error;
  cl_mem sub_buffer = clCreateSubBuffer(buffer, memFlags,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region, &error);
  CHECK_EQ(error, CL_SUCCESS) << "Failed to create sub buffer." << std::endl;
  if (error != CL_SUCCESS) {
    dbgPrint(std::cout << "Failed to create sub buffer ("
                         << error << ")." << std::endl);
    throw(error);
  }
  kernel->arg(argIdx, WrapHandle(sub_buffer, ctx));
  if (preserved)
    subBufferMap.insert(make_pair(std::make_tuple(buffer, offset, size),
                        sub_buffer));
  else
    tmpSubBuffers.push_back(sub_buffer);
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::cleanTmpSubBuffers(
    const MItype *bottom, const MItype *top) {
  for (auto &buffer : tmpSubBuffers)
    clReleaseMemObject(buffer);
  tmpSubBuffers.clear();
}

template<typename MItype>
cl_int LibDNNConvSpatial<MItype>::convolve(
    const MItype *bottom, const MItype *top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  CHECK((!std::is_same<MItype, double>::value));
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  viennacl::ocl::program & program = ctx.get_program(config->kernelName);
  viennacl::ocl::kernel &kernel = program.get_kernel(config->kernelName);
  cl_int err = CL_SUCCESS;

  if (config->kernelType == 2) {
    swizzleWeights(bottom, top, config->workItem_output[2], false);
    size_t total_bottom_size = bottom_dim_ * numImages;
    size_t total_kernel_size = this->kernel_shape_[0]
                             * this->kernel_shape_[1]
                             * this->fmaps_in_ * this->M_FW_;
    size_t total_bias_size = this->M_FW_ * this->group_;
    size_t total_top_size = top_dim_ * numImages;
    for (int_tp g = 0; g < this->group_; ++g) {
      bias_offset_ = this->M_FW_ * g;
      int_tp image_offset = width_ * height_
                                   * (this->fmaps_in_ / this->group_) * g;
      int_tp output_image_offset = output_w_ * output_h_ * this->M_FW_ * g;

      int_tp kernel_offset = this->kernel_shape_[0] * this->kernel_shape_[1]
                             * (this->fmaps_in_ / this->group_)
                             * this->M_FW_ * g;
      cl_uint argIdx = 0;

      try {
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bottom_data_,
                           image_offset,
                           total_bottom_size - image_offset,
                           true, false);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) swizzled_weights_,
                           kernel_offset,
                           total_kernel_size - kernel_offset,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bias_,
                           bias_offset_,
                           total_bias_size - bias_offset_,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) top_data_,
                           output_image_offset,
                           total_top_size - output_image_offset,
                           false, false);
      } catch (int e) {
        err = e;
      }

      if (err == CL_SUCCESS) {
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     kernel.handle().get(), 3,
                                     NULL,
                                     config->global_work_size,
                                     config->local_work_size, 0, NULL,
                                     NULL);
      }
      if (err != CL_SUCCESS)
        break;
    }

    if (this->group_ > 1) {
      cleanTmpSubBuffers(bottom, top);
    }
    if (err != CL_SUCCESS)
      return err;
  } else if (config->kernelType == 5) {
    swizzleWeights(bottom, top, config->workItem_output[1], true);
    size_t total_bottom_size = bottom_dim_ * numImages;
    size_t total_kernel_size = this->kernel_shape_[0]
                             * this->kernel_shape_[1]
                             * this->fmaps_in_ * this->M_FW_;
    size_t total_bias_size = this->M_FW_ * this->group_;
    size_t total_top_size = top_dim_ * numImages;
    for (int_tp g = 0; g < this->group_; ++g) {
      bias_offset_ = this->M_FW_ * g;
      int_tp image_offset = width_ * height_
          * (this->fmaps_in_ / this->group_) * g;
      int_tp output_image_offset = output_w_ * output_h_
          * this->M_FW_ * g;

      cl_uint argIdx = 0;
      int_tp kernel_offset = this->kernel_shape_[0] * this->kernel_shape_[1]
                             * (this->fmaps_in_ / this->group_)
                             * this->M_FW_ * g;
      try {
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bottom_data_,
                           image_offset,
                           total_bottom_size - image_offset,
                           true, false);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) swizzled_weights_,
                           kernel_offset,
                           total_kernel_size - kernel_offset,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bias_,
                           bias_offset_,
                           total_bias_size - bias_offset_,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) top_data_,
                           output_image_offset,
                           total_top_size - output_image_offset,
                           false, false);
      } catch (int e) {
        err = e;
      }

      if (err == CL_SUCCESS) {
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
        err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     kernel.handle().get(), 3,
                                     NULL,
                                     config->global_work_size,
                                     config->local_work_size, 0, NULL,
                                     NULL);
        OCL_CHECK(err);
      }
      if (err != CL_SUCCESS)
        break;
    }

    if (this->group_ > 1) {
      cleanTmpSubBuffers(bottom, top);
    }
    if (err != CL_SUCCESS)
      return err;
  } else {
    for (int_tp N = 0; N < numImages; ++N) {
      for (int_tp g = 0; g < this->group_; ++g) {
        bias_offset_ = this->M_FW_ * g;
        int_tp image_offset = N * this->bottom_dim_
            + width_ * height_ * (this->fmaps_in_ / this->group_) * g;
        int_tp output_image_offset = N * this->top_dim_
            + output_w_ * output_h_ * this->M_FW_ * g;

        cl_uint argIdx = 0;
        int_tp kernel_offset = this->kernel_shape_[0] * this->kernel_shape_[1]
                              * (this->fmaps_in_ / this->group_)
                              * this->M_FW_ * g;

        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom_data_, &ctx));
        kernel.arg(argIdx++, image_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) weight_, &ctx));
        kernel.arg(argIdx++, kernel_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) bias_, &ctx));
        kernel.arg(argIdx++, bias_offset_);
        kernel.arg(argIdx++, WrapHandle((cl_mem) top_data_, &ctx));
        kernel.arg(argIdx++, output_image_offset);
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        kernel.arg(argIdx++, (uint16_t)this->pad_[1]);
        kernel.arg(argIdx++, (uint16_t)this->pad_[0]);
        if (config->use_null_local) {
          err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       kernel.handle().get(), 3,
                                       NULL,
                                       config->global_work_size, NULL, 0, NULL,
                                       NULL);
        } else {
          err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       kernel.handle().get(), 3,
                                       NULL,
                                       config->global_work_size,
                                       config->local_work_size, 0, NULL,
                                       NULL);
        }

        if (err != CL_SUCCESS)
          return err;
      }
    }
  }

  return err;
}

template<typename MItype>
float LibDNNConvSpatial<MItype>::timed_convolve(
    const MItype *bottom, const MItype *top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  // warm up.
  convolve(bottom, top, index, num_, config);
  Timer timer;
  timer.initted();
  timer.Start();
  cl_int err;
  dbgPrint(std::cout << "Bechmarking kernel: " << config->kernelName
           << std::endl);
  err = convolve(bottom, top, index, num_, config);
  timer.Stop();
  if (err != CL_SUCCESS) {
    config->tested = true;
    config->verified = false;
  }

  float elapsedTime = timer.MilliSeconds();
#ifdef dbg
  double out_w = output_w_;
  double out_h = output_h_;
  double out_z = this->M_FW_;
  double k_w = this->kernel_shape_[1];
  double k_h = this->kernel_shape_[0];
  double k_z = this->fmaps_in_;
  double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z)*num_;
  std::cout << "\tEstimated GFLOPS:" << ((totalFlops/1000)/1000)/1000
  << std::endl;
  std::cout << "\tEstimated GFLOPS/s: " <<
  (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime) << std::endl;
#if 0
  std::cout << "Estimated utilization: " <<
  ((((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime))/880.0
  << std::endl;
#endif
#endif
  return elapsedTime;
}

template<typename MItype>
bool LibDNNConvSpatial<MItype>::verify_result(
    const MItype *bottom, const MItype *top,
    int_tp index,
    int_tp numImages, const MItype *verify_blob, kernelConfig* config) {

  uint_tp verificationFail = 0;

  if (config->verified)
    return true;
  else if (config->tested)
    return false;

  if (std::is_same<MItype, half_fp>::value)
    return true;

  greentea_memset(LibDNN<MItype>::dev_ptr_->id(),
                  sizeof(MItype) * numImages * this->top_dim_,
                  0,
                  (cl_mem)top,
                  0);
  config->executionTime = timed_convolve(bottom, top, index, numImages,
                                         config);
  const MItype *verify_data;
  MItype *data;
  MItype *tmp_verify_data;
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  data = reinterpret_cast<MItype *>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(),
        (cl_mem)top, true, CL_MAP_READ,
        0, sizeof(MItype) * numImages * this->top_dim_, 0, NULL, NULL, NULL));
  tmp_verify_data = reinterpret_cast<MItype *>(clEnqueueMapBuffer(
           ctx.get_queue().handle().get(),
           (cl_mem)verify_blob, true, CL_MAP_READ,
           0, sizeof(MItype) * numImages * this->top_dim_,
           0, NULL, NULL, NULL));
  verify_data = tmp_verify_data;

  for (int_tp N = 0; N < numImages; ++N) {
    for (int_tp g = 0; g < this->group_; ++g) {
      int_tp output_image_offset = N * this->top_dim_
          + output_w_ * output_h_ * this->M_FW_ * g;
      for (int_tp out_ch = 0; out_ch < this->M_FW_
                    && !verificationFail; out_ch++)
        for (int_tp h = 0; h < output_h_ && !verificationFail; h++)
          for (int_tp w = 0; w < output_w_; w++) {
            size_t offset = output_image_offset + out_ch * output_w_ * output_h_
                            + h * output_w_ + w;
            if (fabs(data[offset] - verify_data[offset]) >
                       0.1 * fabs(verify_data[offset]) &&
                !(fabs(verify_data[offset]) < 1.e-3
                  && fabs(data[offset] - verify_data[offset]) < 1.e-4)) {
              dbgPrint(printf("test verification failed @ image %d group %d"
                              "out_ch %d h %d w %d got %G expected %G\N",
                      N, g, out_ch, h, w,
                      static_cast<double>(data[offset]),
                      static_cast<double>(verify_data[offset])));
              verificationFail = 1;
              goto out;
            }
          }
    }
  }
out:
  clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
        (cl_mem)top, data, 0, NULL, NULL);
  clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
        (cl_mem)verify_blob, tmp_verify_data, 0, NULL, NULL);
  if (verificationFail == 1)
     return false;
  else
     return true;
}

template<typename MItype>
viennacl::ocl::program LibDNNConvSpatial<MItype>::compile_fw_kernel() {
  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  ctx.build_options(options_);
  return ctx.add_program(LibDNN<MItype>::kernel_.c_str(), kernel_name_);
}

template<typename MItype>
bool LibDNNConvSpatial<MItype>::create_gemm_like_conv_kernel(
    const MItype *bottom, const MItype *top,
    int_tp blockM,
    int_tp blockK, int_tp blockN) {

  int_tp workItemOutput[3] = { blockM, blockK, blockN };
  int_tp output_width = output_w_;
  int_tp output_height = output_h_;
  int_tp simd_size = blockK;
  int_tp num_batches = num_;
  int_tp alignedFilterWidth = ALIGN(this->M_FW_, blockN);
  int_tp alignedExpandHeight = ALIGN(output_width * output_height, blockM);
  int_tp globalWorkSizeDX = blockN;
  int_tp globalWorkSizeDY = blockM;
  size_t sgemm_m = alignedExpandHeight;
  size_t sgemm_n = alignedFilterWidth;
  size_t gx = static_cast<size_t>(ceil(static_cast<MItype>(sgemm_n)
                             / static_cast<MItype>(globalWorkSizeDX)));
  size_t gy = static_cast<size_t>(ceil(static_cast<MItype>(sgemm_m)
                             / static_cast<MItype>(globalWorkSizeDY)));
  gy = ALIGN(gy, blockK);
  size_t gz = num_batches;
  size_t global_size[3] = { gx, gy, gz };
  size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };

  kernelType_ = 5;
  blockM_ = blockM;
  blockK_ = blockK;
  blockN_ = blockN;
  GenerateKernels();
  viennacl::ocl::program program = compile_fw_kernel();

  size_t workgroupSize_used;
  viennacl::ocl::kernel & kernel = program.get_kernel(kernel_name_);
  cl_int err = clGetKernelWorkGroupInfo(
      kernel.handle().get(), viennacl::ocl::current_device().id(),
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &workgroupSize_used,
      NULL);

  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  if (workgroupSize_used != simd_size) {
    ctx.delete_program(kernel_name_);
    return false;
  }

  if (err == CL_SUCCESS || err == true) {
    kernelQueue.push_back(
        new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                         false, true, false, 5));
    return true;
  } else {
    ctx.delete_program(kernel_name_);
    return false;
  }
}

template<typename MItype>
bool LibDNNConvSpatial<MItype>::setup_IDLF(
    const MItype *bottom, const MItype *top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp simd_size) {
  int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };
  const int_tp num_output_maps = this->M_FW_;
  int_tp output_width = output_w_;
  int_tp output_height = output_h_;
  int_tp output_block_width = blockWidth;
  int_tp output_block_height = blockHeight;
  int_tp num_batches = num_;

  size_t global_size[3] = { (size_t) (output_width + output_block_width - 1)
      / output_block_width, (size_t) (output_height + output_block_height - 1)
      / output_block_height,
      (size_t) num_batches *
      ALIGN(num_output_maps, simd_size) };
  size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

  kernelType_ = KERNEL_TYPE_INTEL_IDLF;
  blockM_ = blockWidth;
  blockK_ = blockHeight;
  blockN_ = simd_size;

  GenerateKernels();
  viennacl::ocl::program program = compile_fw_kernel();

  // ClKernel kernel;
  size_t workgroupSize_used;
  viennacl::ocl::kernel &kernel = program.get_kernel(kernel_name_);
  cl_int err = clGetKernelWorkGroupInfo(
      kernel.handle().get(), viennacl::ocl::current_device().id(),
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &workgroupSize_used,
      NULL);

  viennacl::ocl::context &ctx =
     viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
  if (workgroupSize_used != simd_size) {
    ctx.delete_program(kernel_name_);
    return false;
  }

  if (err == CL_SUCCESS || err == true) {
    kernelQueue.push_back(
        new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                         false, true, false, 2));
    return true;
  } else {
    ctx.delete_program(kernel_name_);
    return false;
  }
}

template<typename MItype>
bool LibDNNConvSpatial<MItype>::tune_local_size(
    const MItype *bottom, const MItype *top,
    kernelConfig* config) {
  if (config->use_null_local || !config->autoTune)
    return true;

  float fastestTime = 999999990000000000000000000.0f;
  uint_tp multiplier = 4;
  uint_tp localSize[3] = { 1, 1, 1 };

  int_tp skip = 0;
  Timer timer;
  timer.initted();
  bool allFailed = true;
  for (int_tp z = 0; z <= 16; z++) {
    for (int_tp Y = 0; Y <= 16; Y++) {
      for (int_tp X = 1; X <= 16; X++) {
        timer.Start();
        skip = 0;

        if (config->autoTune) {
          config->local_work_size[0] =
              (multiplier * X == 0) ? 1 : multiplier * X;
          config->local_work_size[1] =
              (multiplier * Y == 0) ? 1 : multiplier * Y;
          config->local_work_size[2] =
              (multiplier * z == 0) ? 1 : multiplier * z;

          calculate_global_size(1, config->workItem_output,
                                config->local_work_size,
                                config->global_work_size);
        }
        if (config->workItem_output[2] *
            config->global_work_size[2] != this->M_FW_)
          break;

        if (config->swizzle_weights) {
          z = 32;
        }

        int err = 0;
        err = convolve(bottom, top, 0, 1, config);

        if (err != CL_SUCCESS)
          skip = 1;

        if (skip) {
          timer.Stop();
          break;
        }
        timer.Stop();
        allFailed = false;
        float elapsedTime = timer.MilliSeconds();

        if (elapsedTime < fastestTime) {
          fastestTime = elapsedTime;
          localSize[0] = config->local_work_size[0];
          localSize[1] = config->local_work_size[1];
          localSize[2] = config->local_work_size[2];
        }
      }
    }
  }
  if (allFailed) {
    // 1,1,1 is never a good local size and no need to test at all.
    dbgPrint(std::cout << "Can't find good local size for "
                       << config->kernelName << std::endl);
    return false;
  }

  dbgPrint(std::cout << "Best local size[" << localSize[0] << "][" <<
      localSize[1] << "]["<< localSize[2] << "]: " << fastestTime
      << " Kernel_h: " << this->kernel_shape_[0]
      << " Kernel_w: " << this->kernel_shape_[1]
      << " Stride_h: " << this->stride_[1]
      << " Stride_w: " << this->stride_[1]
      << " Pad_h: " << this->pad_[1]
      << " Pad_w: " << this->pad_[1] << std::endl);

  if (config->autoTune) {
    for (int_tp li = 0; li < 3; li++)
      config->local_work_size[li] = localSize[li];

    calculate_global_size(1, config->workItem_output, config->local_work_size,
                          config->global_work_size);
  }
  return true;
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::create_convolution_kernel(
    const MItype *bottom, const MItype *top,
    int_tp kernelType,
    int_tp blockWidth, int_tp blockHeight,
    int_tp blockDepth) {
  if (kernelType == 2)
    setup_IDLF(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 4)
    create_basic_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 5)
    create_gemm_like_conv_kernel(bottom, top,
          blockWidth, blockHeight, blockDepth);
  else
    assert(0);
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::setup_convolution(
    const MItype *bottom, const MItype *top,
    const MItype *verify_blob) {
  // Initializes unique kernel ID
  kernel_uid_ = 0;

  if (LibDNN<MItype>::dev_ptr_->CheckCapability("cl_intel_subgroups")) {
    /* IDLF kernels are using Intel specific extension which make
       them intel only. */
    // Generates static key_
    viennacl::ocl::context &ctx =
       viennacl::ocl::get_context(LibDNN<MItype>::dev_ptr_->id());
    int_tp max_compute_units = ctx.current_device().max_compute_units();
    int_tp kernelCnt = 0;
    if (this->group_ == 1
        && ((this->M_FW_ % 8 == 0)
        && (this->M_FW_ % 32 != 24))) {
      create_convolution_kernel(bottom, top, 5, 1, 8, 32);
      create_convolution_kernel(bottom, top, 5, 2, 8, 32);
      if ((this->kernel_shape_[1] < 4 ||
           (std::is_same<MItype, half_fp>::value))
          && this->M_FW_ % 32 == 0)
        create_convolution_kernel(bottom, top, 5, 1, 16, 32);
    }

    for (int_tp simd_size = 8; simd_size <= 16; simd_size += 8) {
      if (simd_size == 8
          && !((this->group_ == 1 || this->M_FW_ % 8 == 0)))
        continue;
      if (simd_size == 16
          && !(this->group_ == 1 || this->M_FW_ % 16 == 0))
        continue;
      int_tp width_max, height_max, block_size_max;
      if (simd_size == 8) {
        width_max = 16;
        height_max = 16;
        block_size_max = 64;
      } else {
        width_max = 14;
        height_max = 14;
        block_size_max = 32;
      }
      for (uint32_t width = width_max; width > 0; width--) {
        int_tp candidate = 0;
        if (width > output_w_)
          continue;
        for (uint32_t height = height_max; height > 0; height--) {
          if (width * height > block_size_max || height > output_h_)
            continue;
          // Only when the work items count is less than the device
          // max work items or the this->M_FW_ is less than 16, we will tune
          // for simd 8.
          if (simd_size == 8
              && this->M_FW_ >= 16
              && ((num_ * this->M_FW_ * output_w_ * output_h_ /
                   static_cast<MItype>(width * height))
                 >= max_compute_units * 7 * 16))
            continue;
          int_tp tile_x = (this->kernel_shape_[1] * this->dilation_[1]
                       + (width - 1) * this->stride_[1] + 3) & ~3;
          int_tp tile_y = this->kernel_shape_[0]
                     * this->dilation_[0] + (height - 1)
                     * this->stride_[0];
          if (tile_x > (4 * simd_size))
            continue;
          int_tp tile_y_stride = (4 * simd_size) / tile_x;

          if ((tile_y + tile_y_stride - 1) / tile_y_stride < 4) {
            create_convolution_kernel(bottom, top, 2, width, height, simd_size);
            candidate++;
          }
          if (candidate >= 4 && height == 2)
            break;
        }
        kernelCnt += candidate;
        if (kernelCnt >= 12 && width == 2)
          break;
      }
    }
  }
  for (int_tp X = 0; X < kernelQueue.size(); X++) {
    if (tune_local_size(bottom, top, kernelQueue[X])) {
      kernelQueue[X]->executionTime = timed_convolve(bottom, top, bottom_index_,
                                                     num_, kernelQueue[X]);
    } else {
      // skip those kernels without a good local size.
      kernelQueue[X]->verified = false;
      kernelQueue[X]->tested = true;
    }
#ifdef TEST_ALL_KERNELS
    if (kernelQueue[X]->tested == false) {
      bool verified = verify_result(bottom, top, bottom_index_, num_,
                                      verify_blob, kernelQueue[X]);
      if (verified == false) {
        dbgPrint(std::cout << "Kernel "
                             << kernelQueue[X]->kernelName
                             << " failed verification" << std::endl);
        dbgPrint(std::cout << "kernelQueue[X]->workItem_output[0]: "
                       << kernelQueue[X]->workItem_output[0] << " "
                       << "kernelQueue[X]->workItem_output[1]: "
                       << kernelQueue[X]->workItem_output[1] << " "
                       << "kernelQueue[X]->workItem_output[2]: "
                       << kernelQueue[X]->workItem_output[2] << " "
                       << "kernelQueue[X]->kernelType: "
                       << kernelQueue[X]->kernelType << " "
                       << "kernelQueue[X]->global_work_size[0]: "
                       << kernelQueue[X]->global_work_size[0] << " "
                       << "kernelQueue[X]->global_work_size[1]: "
                       << kernelQueue[X]->global_work_size[1] << " "
                       << "kernelQueue[X]->global_work_size[2]: "
                       << kernelQueue[X]->global_work_size[2] << " "
                       << "kernelQueue[X]->local_work_size[0]: "
                       << kernelQueue[X]->local_work_size[0] << " "
                       << "kernelQueue[X]->local_work_size[1]: "
                       << kernelQueue[X]->local_work_size[1] << " "
                       << "kernelQueue[X]->local_work_size[2]: "
                       << kernelQueue[X]->local_work_size[2] << " "
                       << kernelQueue[X]->swizzle_weights << " "
                       << kernelQueue[X]->use_null_local << std::endl);
      } else {
        dbgPrint(std::cout << "Kernel "
                           << kernelQueue[X]->kernelName
                           << " pass verification" << std::endl);
      }
    }
#endif
  }
  int_tp failures = 0;
  bool verification = false;
  if (kernelQueue.size()) {
    while (failures < kernelQueue.size()) {
      int_tp fastestKernel = -1;
      float fastestTime = 999999990000000000000000000.0f;

      for (int_tp X = 0; X < kernelQueue.size(); X++) {
        if (kernelQueue[X]->executionTime < fastestTime
            && kernelQueue[X]->tested == false) {
          fastestKernel = X;
          fastestTime = kernelQueue[X]->executionTime;
        }
      }
      if (fastestKernel < 0) break;
      // Test fastest kernel
      bool verified = verify_result(bottom, top, bottom_index_, num_,
                                    verify_blob, kernelQueue[fastestKernel]);
      if (verified == true) {
        kernelQueue[fastestKernel]->verified = true;
        kernel_index_ = fastestKernel;
        verification = true;
        break;
      } else {
        kernelQueue[fastestKernel]->tested = true;
        dbgPrint(std::cout << "Kernel "
                           << kernelQueue[fastestKernel]->kernelName
                           << " failed verification" << std::endl);
        failures++;
      }
    }
  }
  if (verification) {
    dbgPrint(std::cout << "Kernel <" << kernelQueue[kernel_index_]->kernelName
                       << "> passed verification" << std::endl);
  } else {
    dbgPrint(std::cout << "Verification was not successful, "
                       << "fallback to basic kernel" << std::endl);
    create_basic_kernel(bottom, top, 1, 1, 1);
    kernel_index_ = kernelQueue.size() - 1;
    verification = verify_result(bottom, top, bottom_index_, num_,
                                 verify_blob, kernelQueue[kernel_index_]);
    CHECK_EQ(verification, true) << "Basic kernel failed verification."
                                 << std::endl;
  }
  this->bestKernelConfig = kernelQueue[kernel_index_];

  dbgPrint(std::cout << "Convolution Time:"
                     << kernelQueue[kernel_index_]->executionTime << std::endl);

  if (bestKernelConfig->kernelType != 2 && bestKernelConfig->kernelType != 5)
    swizzled_weights_ = NULL;

  for (int_tp X = 0; X < kernelQueue.size(); X++) {
    if (X != kernel_index_) {
      viennacl::ocl::current_context().delete_program(
          kernelQueue[X]->kernelName);
      delete kernelQueue[X];
    }
  }
  kernelQueue.clear();

  tuned_ = true;

  string outputFile;
  outputFile = cache_path_.str() + key_;
  std::ifstream cachedKernel(outputFile.c_str());
  std::ofstream outputKernel;
  outputKernel.open(outputFile.c_str());
  outputKernel << bestKernelConfig->workItem_output[0] << " "
               << bestKernelConfig->workItem_output[1] << " "
               << bestKernelConfig->workItem_output[2] << " "
               << bestKernelConfig->kernelType << " "
               << bestKernelConfig->global_work_size[0] << " "
               << bestKernelConfig->global_work_size[1] << " "
               << bestKernelConfig->global_work_size[2] << " "
               << bestKernelConfig->local_work_size[0] << " "
               << bestKernelConfig->local_work_size[1] << " "
               << bestKernelConfig->local_work_size[2] << " "
               << bestKernelConfig->swizzle_weights << " "
               << 0 << " "  // deprecated
               << bestKernelConfig->use_null_local << " ";
  outputKernel.close();
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::load_cached_kernels(
    const MItype *bottom, const MItype *top) {
  // Generates static key_
  string previous_key = key_;
  generate_key();
  int_tp prev_kernel_type = 0;
  if (tuned_) {
    if (key_.compare(previous_key) == 0)
      return;
    tuned_ = false;
    prev_kernel_type = bestKernelConfig->kernelType;
    viennacl::ocl::current_context().
      delete_program(bestKernelConfig->kernelName);
    delete bestKernelConfig;
    bestKernelConfig = NULL;
  }
  // Initializes unique kernel ID
  kernel_uid_ = 0;

  // Find cached kernel configuration
  string outputFile;
  outputFile = cache_path_.str() + key_;
  std::ifstream cachedKernel(outputFile.c_str());
  if (cachedKernel) {
    int_tp X, Y, z, type;
    cachedKernel >> X;
    cachedKernel >> Y;
    cachedKernel >> z;
    cachedKernel >> type;
    if (type == 2) {
      if (z == 1)
        z = 16;
      CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
    }
    create_convolution_kernel(bottom, top, type, X, Y, z);
    kernel_index_ = kernelQueue.size() - 1;
    if (kernel_index_ == -1) {
      std::cerr << "Failed to get kernel from cached configurations."
                << std::endl;
      std::cerr << "Deleting broken cache file and try tuning again..."
                << std::endl;
      string bakFile = outputFile + ".bak";
      std::rename(outputFile.c_str(), bakFile.c_str());
      return;
    }
    bestKernelConfig = kernelQueue[kernel_index_];
    kernelQueue.clear();
    // As we are using varying image size kernels now, let's skip the
    // cached work group size and local group size here, and we already
    // get correct work/local group size at the create_convolution kernel stage.
    // To not break the previous trained record, for now just skipping them.
    // Will use a totally different cache mechanism in the future.
    size_t foo;  // for deprecated parameters.
    cachedKernel >> foo;
    cachedKernel >> foo;
    cachedKernel >> foo;
    cachedKernel >> bestKernelConfig->local_work_size[0];
    cachedKernel >> bestKernelConfig->local_work_size[1];
    cachedKernel >> bestKernelConfig->local_work_size[2];
    if (bestKernelConfig->kernelType == 1)
      calculate_global_size(1, bestKernelConfig->workItem_output,
                            bestKernelConfig->local_work_size,
                            bestKernelConfig->global_work_size);
    cachedKernel >> bestKernelConfig->swizzle_weights;
    cachedKernel >> foo;
    cachedKernel >> bestKernelConfig->use_null_local;
    tuned_ = true;
    // If kernel type changed to type 2 or 4, we need to reset the swizzled
    // weights pointer to invalidate the previous swizzled weights data.
    if (prev_kernel_type != bestKernelConfig->kernelType &&
        (bestKernelConfig->kernelType == 2 ||
         bestKernelConfig->kernelType == 5))
      swizzled_weights_ = NULL;
  }
  return;
}

template<typename MItype>
void LibDNNConvSpatial<MItype>::SetUp(
    const MItype *bottom, const MItype *top,
    caffe::Backend backend) {
  if (backend == caffe::BACKEND_OPENCL) {
    load_cached_kernels(bottom, top);
  }
}

INSTANTIATE_CLASS(LibDNNConvSpatial);

}  // namespace caffe
#endif  // USE_OPENCL
#endif  // USE_INTEL_SPATIAL
