#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
vector<int> LocalConvolutionLayer<Dtype>::compute_output_shape() const {
  const int *kernel_shape_data = this->kernel_shape_.cpu_data();
  const int *stride_data = this->stride_.cpu_data();
  const int *pad_data = this->pad_.cpu_data();
  const int *dilation_data =
      this->dilation_.cpu_data(); // not use dilation here in fact
  vector<int> output_shape;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int *input_dim = this->conv_input_shape_.cpu_data();
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim[i + 1] + 2 * pad_data[i] - kernel_extent) / stride_data[i] +
        1;
    output_shape.push_back(output_dim);
  }
  return output_shape;
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::init_local_offset() {
  int h, w, offset_h, offset_w, symmetry_offset_h, symmetry_offset_w;
  Blob<int> &idx_to_off = this->loc_idx_to_offset_;
  int *idx_to_off_data = idx_to_off.mutable_cpu_data();
  int loc_h = this->conv_input_shape_.cpu_data()[1];
  int loc_w = this->conv_input_shape_.cpu_data()[2];
  for (h = 0; h < this->local_region_num_h_ / 2; ++h) {
    offset_h = h * this->local_region_step_h_;
    symmetry_offset_h = this->bottom_height_ - (offset_h + loc_h);
    for (w = 0; w < this->local_region_num_w_ / 2; ++w) {
      offset_w = w * this->local_region_step_w_;
      symmetry_offset_w = this->bottom_width_ - (offset_w + loc_w);
      idx_to_off_data[idx_to_off.offset(h, w, 0, 0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, w, 1, 0)] = offset_w;

      idx_to_off_data[idx_to_off.offset(h, this->local_region_num_w_ - 1 - w, 0,
                                        0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, this->local_region_num_w_ - 1 - w, 1,
                                        0)] = symmetry_offset_w;

      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h, w, 0,
                                        0)] = symmetry_offset_h;
      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h, w, 1,
                                        0)] = offset_w;

      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h,
                                        this->local_region_num_w_ - 1 - w, 0,
                                        0)] = symmetry_offset_h;
      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h,
                                        this->local_region_num_w_ - 1 - w, 1,
                                        0)] = symmetry_offset_w;
    }
    if (local_region_num_w_ % 2) {
      offset_w = (this->bottom_width_ - loc_w) / 2;

      idx_to_off_data[idx_to_off.offset(h, w, 0, 0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, w, 1, 0)] = offset_w;

      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h, w, 0,
                                        0)] = symmetry_offset_h;
      idx_to_off_data[idx_to_off.offset(this->local_region_num_h_ - 1 - h, w, 1,
                                        0)] = offset_w;
    }
  }
  if (this->local_region_num_h_ % 2) {
    offset_h = (this->bottom_height_ - loc_h) / 2;
    for (w = 0; w < this->local_region_num_w_ / 2; ++w) {
      offset_w = w * this->local_region_step_w_;
      symmetry_offset_w = this->bottom_width_ - (offset_w + loc_w);

      idx_to_off_data[idx_to_off.offset(h, w, 0, 0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, w, 1, 0)] = offset_w;

      idx_to_off_data[idx_to_off.offset(h, this->local_region_num_w_ - 1 - w, 0,
                                        0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, this->local_region_num_w_ - 1 - w, 1,
                                        0)] = symmetry_offset_w;
    }
    if (this->local_region_num_w_ % 2) {
      offset_w = (this->bottom_width_ - loc_w) / 2;
      idx_to_off_data[idx_to_off.offset(h, w, 0, 0)] = offset_h;
      idx_to_off_data[idx_to_off.offset(h, w, 1, 0)] = offset_w;
    }
  }
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::realign_loc_conv_result_cpu(
    const Dtype *local_conv_data, Dtype *dst_data) {
  int num_output = this->num_output_;
  auto output_shape = compute_output_shape();
  int height_out = output_shape[0], width_out = output_shape[1];
  int top_height = this->top_height_, top_width = this->top_width_;
  int local_region_num_w = this->local_region_num_w_;

  int mStep = top_width;
  int loc_conv_res_size = this->num_output_ * height_out * width_out;
  for (int n = 0; n < num_output; ++n) {
    int num_offset = n * height_out * width_out;
    for (int h = 0; h < top_height; ++h) {
      for (int w = 0; w < top_width; ++w) {
        int dst_offset = h * top_width + w;
        int dst_idx = dst_offset + n * top_height * top_width;
        int loc_w = dst_offset % mStep % width_out;
        int loc_idx_w = dst_offset % mStep / width_out;
        int loc_h = dst_offset / mStep % height_out;
        int loc_idx_h = dst_offset / mStep / height_out;

        int loc_data_offset = num_offset + loc_h * width_out + loc_w;
        int loc_offset =
            (loc_idx_h * local_region_num_w + loc_idx_w) * loc_conv_res_size;
        int src_idx = loc_offset + loc_data_offset;
        dst_data[dst_idx] = local_conv_data[src_idx];
      }
    }
  }
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::crop_loc_patch_cpu(
    const Dtype *src, int src_w, int src_h, int src_c, int crop_width,
    int crop_height, int w_off, int h_off, Dtype *local_patch_data) {
  for (int c = 0; c < src_c; ++c) {
    for (int h = 0; h < crop_height; ++h) {
      for (int w = 0; w < crop_width; ++w) {
        local_patch_data[(c * crop_height + h) * crop_width + w] =
            src[(c * src_h + (h + h_off)) * src_w + w + w_off];
      }
    }
  }
}

/*
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::realign_bottom_diff_cpu(const Dtype
*loc_bottom_diff_buffer, Dtype *bottom_diff)
{
        const Blob<int> *idx_to_off_blob = &this->loc_idx_to_offset_;
        const int *idx_to_off_data = idx_to_off_blob->cpu_data();

        const Dtype *src_data = loc_bottom_diff_buffer;
        Dtype *dst_data = bottom_diff;

        int loc_height = this->conv_input_shape_.cpu_data()[1], loc_width =
this->conv_input_shape_.cpu_data()[2]; int bottom_width = this->bottom_width_;
        int src_spatial_dim = loc_height * loc_width;
        int src_step = this->conv_in_channels_ * src_spatial_dim;
        int dst_channel_step = this->bottom_height_ * this->bottom_width_;
        int channels = this->channels_;
        int loc_num_w = this->local_region_num_w_, loc_num_h =
this->local_region_num_h_; for (int n = 0; n < channels; n++){ for (int lh = 0;
lh < loc_num_h; lh++){ for (int lw = 0; lw < loc_num_w; lw++){ int loc_off_h =
idx_to_off_data[idx_to_off_blob->offset(lh, lw, 0, 0)]; int loc_off_w =
idx_to_off_data[idx_to_off_blob->offset(lh, lw, 1, 0)];

                                int loc_num = lh * loc_num_w + lw;
                                int src_offset = loc_num * src_step + n *
src_spatial_dim; for (int h = 0; h < loc_height; h++){ for (int w = 0; w <
loc_width; w++){ int dst_idx = (loc_off_h + h) * bottom_width + loc_off_w + w;
                                                int src_idx = src_offset + h *
loc_width + w; dst_data[dst_idx] += src_data[src_idx];
                                        }
                                }

                        }
                }
                dst_data += dst_channel_step;
        }
}
*/
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(4, bottom[0]->num_axes())
      << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  LocalConvolutionParameter loc_conv_param =
      this->layer_param_.local_conv_param();
  CHECK(!loc_conv_param.has_kernel_size() !=
        !(loc_conv_param.has_kernel_h() && loc_conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(loc_conv_param.has_kernel_size() ||
        (loc_conv_param.has_kernel_h() && loc_conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!loc_conv_param.has_pad() && loc_conv_param.has_pad_h() &&
         loc_conv_param.has_pad_w()) ||
        (!loc_conv_param.has_pad_h() && !loc_conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!loc_conv_param.has_stride() && loc_conv_param.has_stride_h() &&
         loc_conv_param.has_stride_w()) ||
        (!loc_conv_param.has_stride_h() && !loc_conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  CHECK((!loc_conv_param.has_local_region_number() &&
         loc_conv_param.has_local_region_number_w() &&
         loc_conv_param.has_local_region_number_h()) ||
        (!loc_conv_param.has_local_region_number_h() &&
         !loc_conv_param.has_local_region_number_w()))
      << "has_local_region_number is local number OR has_local_region_number_h "
         "and has_local_region_number_w are required.";
  CHECK((!loc_conv_param.has_local_region_ratio() &&
         loc_conv_param.has_local_region_ratio_w() &&
         loc_conv_param.has_local_region_ratio_h()) ||
        (!loc_conv_param.has_local_region_ratio_w() &&
         !loc_conv_param.has_local_region_ratio_h()))
      << "has_local_region_ratio is local number OR has_local_region_ratio_h "
         "and has_local_region_ratio_w are required.";
  CHECK((!loc_conv_param.has_local_region_step() &&
         loc_conv_param.has_local_region_step_w() &&
         loc_conv_param.has_local_region_step_h()) ||
        (!loc_conv_param.has_local_region_step_w() &&
         !loc_conv_param.has_local_region_step_h()))
      << "has_local_region_step is local_region_step OR "
         "has_local_region_step_w and has_local_region_step_h are required.";

  // Configure the local region number, size and step.
  if (loc_conv_param.has_local_region_number()) {
    local_region_num_w_ = local_region_num_h_ =
        loc_conv_param.local_region_number();
  } else {
    local_region_num_w_ = loc_conv_param.local_region_number_w();
    local_region_num_h_ = loc_conv_param.local_region_number_h();
  }
  if (loc_conv_param.has_local_region_ratio()) {
    local_region_ratio_w_ = local_region_ratio_h_ =
        loc_conv_param.local_region_ratio();
  } else {
    local_region_ratio_w_ = loc_conv_param.local_region_ratio_w();
    local_region_ratio_h_ = loc_conv_param.local_region_ratio_h();
  }
  if (loc_conv_param.has_local_region_step()) {
    local_region_step_w_ = local_region_step_h_ =
        loc_conv_param.local_region_step();
  } else {
    local_region_step_w_ = loc_conv_param.local_region_step_w();
    local_region_step_h_ = loc_conv_param.local_region_step_h();
  }

  this->force_nd_im2col_ = false;
  this->num_spatial_axes_ = 2; // height, width for 2D convolution
  this->channel_axis_ = 1;
  vector<int> spatial_dim_blob_shape(1, this->num_spatial_axes_);
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int *kernel_shape_data = this->kernel_shape_.mutable_cpu_data();

  // Configure the kernel size, padding, stride, and inputs.
  if (loc_conv_param.has_kernel_size()) {
    kernel_shape_data[0] = kernel_shape_data[1] = loc_conv_param.kernel_size();
  } else {
    kernel_shape_data[0] = loc_conv_param.kernel_h();
    kernel_shape_data[1] = loc_conv_param.kernel_w();
  }
  CHECK_GT(kernel_shape_data[0], 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_shape_data[1], 0) << "Filter dimensions cannot be zero.";
  this->pad_.Reshape(spatial_dim_blob_shape);
  int *pad_data = this->pad_.mutable_cpu_data();
  if (!loc_conv_param.has_pad_h()) {
    pad_data[0] = pad_data[1] = loc_conv_param.pad();
  } else {
    pad_data[0] = loc_conv_param.pad_h();
    pad_data[1] = loc_conv_param.pad_w();
  }
  this->stride_.Reshape(spatial_dim_blob_shape);
  int *stride_data = this->stride_.mutable_cpu_data();
  if (!loc_conv_param.has_stride_h()) {
    stride_data[0] = stride_data[1] = loc_conv_param.stride();
  } else {
    stride_data[0] = loc_conv_param.stride_h();
    stride_data[1] = loc_conv_param.stride_w();
  }

  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int *dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = loc_conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] =
        (num_dilation_dims == 0)
            ? kDefaultDilation
            : loc_conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = kernel_shape_data[0] == 1 && kernel_shape_data[1] == 1 &&
                  stride_data[0] == 1 && stride_data[1] == 1 &&
                  pad_data[0] == 0 && pad_data[1] == 0;
  // Configure output channels and groups.
  this->channels_ = bottom[0]->channels();
  this->num_output_ = loc_conv_param.num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = loc_conv_param.group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";

  this->conv_out_channels_ = this->num_output_;
  // this->conv_in_channels_ = this->channels_;
  this->L_ = this->local_region_num_w_ *
             this->local_region_num_h_; // number of local regions

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  this->bias_term_ = loc_conv_param.bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        this->L_, this->conv_out_channels_ * this->channels_ / this->group_,
        kernel_shape_data[0], kernel_shape_data[1]));
    shared_ptr<Filler<Dtype>> weight_filler(
        GetFiller<Dtype>(loc_conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(this->L_, this->num_output_, 1, 1));
      shared_ptr<Filler<Dtype>> bias_filler(
          GetFiller<Dtype>(loc_conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  this->loc_idx_to_offset_.Reshape(this->local_region_num_h_,
                                   this->local_region_num_w_, 2, 1);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom[0]->channels(), this->channels_)
      << "Input size incompatible with"
         " weights.";
  CHECK_EQ(4, bottom[0]->num_axes())
      << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";

  const int num = bottom[0]->num();
  this->bottom_height_ = bottom[0]->height();
  this->bottom_width_ = bottom[0]->width();

  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(this->channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(this->bottom_height_, bottom[0]->height())
        << "Inputs must have same height.";
    CHECK_EQ(this->bottom_width_, bottom[0]->width())
        << "Inputs must have same width.";
  }
  // local region height and width
  vector<int> conv_input_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(conv_input_dim_blob_shape);
  int *conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();

  conv_input_shape_data[0] = this->channels_;
  conv_input_shape_data[1] =
      static_cast<int>(bottom_height_ * local_region_ratio_h_);
  conv_input_shape_data[2] =
      static_cast<int>(bottom_width_ * local_region_ratio_w_);

  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  auto output_shape = compute_output_shape();
  this->top_height_ = output_shape[0] * this->local_region_num_h_;
  this->top_width_ = output_shape[1] * this->local_region_num_w_;

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num, this->num_output_, this->top_height_,
                         this->top_width_);
  }
  this->conv_out_spatial_dim_ = output_shape[0] * output_shape[1];

  this->kernel_dim_ = this->channels_ * this->kernel_shape_.cpu_data()[0] *
                      this->kernel_shape_.cpu_data()[1] / this->group_;

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.

  this->col_buffer_.Reshape(1, this->kernel_dim_ * this->group_,
                            output_shape[0], output_shape[1]);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, output_shape[0] * output_shape[1]);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
              this->bias_multiplier_.mutable_cpu_data());
  }

  // create map of local region index to local region offset, the local regions'
  // offsets are central symmetry
  //  int h, w, offset_w, symmetry_offset_w, offset_h, symmetry_offset_h;
  init_local_offset();

  loc_bottom_buffer_.Reshape(this->L_, conv_input_shape_data[0],
                             conv_input_shape_data[1],
                             conv_input_shape_data[2]);
  loc_top_buffer_.Reshape(this->L_, this->conv_out_channels_, output_shape[0],
                          output_shape[1]);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Dtype *loc_bottom_data = loc_bottom_buffer_.mutable_cpu_data();
  Dtype *loc_top_data = loc_top_buffer_.mutable_cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();

  const Blob<int> *idx_to_off = &this->loc_idx_to_offset_;
  const int *idx_to_off_data = idx_to_off->cpu_data();
  const int num = bottom[0]->num();

  for (int i = 0; i < bottom.size(); i++) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    int bottom_w = bottom[i]->width();
    int bottom_h = bottom[i]->height();
    int bottom_c = bottom[i]->channels();
    Dtype *top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < num; n++) {
      const Dtype *single_bottom_data = bottom_data + bottom[i]->offset(n);
      for (int lh = 0; lh < local_region_num_h_; lh++) {
        for (int lw = 0; lw < local_region_num_w_; lw++) {
          int loc_num = lh * local_region_num_w_ + lw;
          const Dtype *loc_weight = weight + this->blobs_[0]->offset(loc_num);
          Dtype *loc_bottom =
              loc_bottom_data + loc_bottom_buffer_.offset(loc_num);
          Dtype *loc_top = loc_top_data + loc_top_buffer_.offset(loc_num);
          crop_loc_patch_cpu(single_bottom_data, bottom_w, bottom_h, bottom_c,
                             this->conv_input_shape_.cpu_data()[2],
                             this->conv_input_shape_.cpu_data()[1],
                             idx_to_off_data[idx_to_off->offset(lh, lw, 1, 0)],
                             idx_to_off_data[idx_to_off->offset(lh, lw, 0, 0)],
                             loc_bottom);
          this->forward_cpu_gemm(loc_bottom, loc_weight, loc_top, false);
          if (this->bias_term_) {
            const Dtype *bias =
                this->blobs_[1]->cpu_data() + this->blobs_[1]->offset(loc_num);
            this->forward_cpu_bias(loc_top, bias);
          }
        }
      }
      realign_loc_conv_result_cpu(loc_top_data, top_data + top[i]->offset(n));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LocalConvolutionLayer);
#endif

INSTANTIATE_CLASS(LocalConvolutionLayer);
REGISTER_LAYER_CLASS(LocalConvolution);
} // namespace caffe
