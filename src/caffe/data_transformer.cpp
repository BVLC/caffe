#ifndef OSX
#include <opencv2/core/core.hpp>
#endif

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param)
    : param_(param) {
  phase_ = Caffe::phase();
  ResetState();
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == Caffe::TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::ResetState() {
  state_.persistent = param_.persistent();
  state_.reset = true;
  state_.do_mirror = false;
  state_.h_off = 0;
  state_.w_off = 0;
  state_.input_channels = 0;
  state_.input_height = 0;
  state_.input_width = 0;
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  const int input_channels = datum.channels();
  const int input_height = datum.height();
  const int input_width = datum.width();

  const int output_channels = transformed_blob->channels();
  const int output_height = transformed_blob->height();
  const int output_width = transformed_blob->width();

  CheckSizes(input_channels, input_height, input_width,
    output_channels, output_height, output_width);

  CHECK_GE(transformed_blob->num(), 1);

  vector<const uchar*> data_ptrs (1, NULL);
  data_ptrs[0] = reinterpret_cast<const uchar*>(datum.data().c_str());
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  // For Datum use:
  const int height_offset = input_width;
  const int channel_offset = input_height * input_width;

  UpdateState(input_channels, input_height, input_width);
  InternalTransform(data_ptrs, height_offset, channel_offset,
    output_channels, output_height, output_width, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be smaller than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifndef OSX
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int input_channels = cv_img.channels();
  const int input_height = cv_img.rows;
  const int input_width = cv_img.cols;

  const int output_channels = transformed_blob->channels();
  const int output_height = transformed_blob->height();
  const int output_width = transformed_blob->width();

  CheckSizes(input_channels, input_height, input_width,
    output_channels, output_height, output_width);

  CHECK_GE(transformed_blob->num(), 1);
  CHECK(cv_img.data) << "Image without data";

  // For cv::Mat use:
  const int num_blocks = cv_img.isContinuous()? 1 : input_height;
  const int height_offset = input_width;
  const int channel_offset = 1;

  UpdateState(input_channels, input_height, input_width);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  if (cv_img.depth() == CV_8U) {
    // Image data is unsigned byte
    vector<const uchar*> data_ptrs (num_blocks, NULL);
    for (int i = 0; i < num_blocks; ++i) {
      data_ptrs[i] = cv_img.ptr<uchar>(i);
    }
    InternalTransform(data_ptrs, height_offset, channel_offset,
    output_channels, output_height, output_width, transformed_data);
  }
  if (cv_img.depth() == CV_8S) {
    // Image data is signed byte
    vector<const char*> data_ptrs (num_blocks, NULL);
    for (int i = 0; i < num_blocks; ++i) {
      data_ptrs[i] = cv_img.ptr<char>(i);
    }
    InternalTransform(data_ptrs, height_offset, channel_offset,
    output_channels, output_height, output_width, transformed_data);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & cv_img_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int cv_img_num = cv_img_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(cv_img_num, 0) << "There is no datum to add";
  CHECK_LE(cv_img_num, num) <<
    "The size of cv_img_vector must be smaller than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < cv_img_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(cv_img_vector[item_id], &uni_blob);
  }
}
#endif

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Blob<Dtype>& input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob.num();
  const int input_channels = input_blob.channels();
  const int input_height = input_blob.height();
  const int input_width = input_blob.width();

  const int output_num = transformed_blob->num();
  const int output_channels = transformed_blob->channels();
  const int output_height = transformed_blob->height();
  const int output_width = transformed_blob->width();

  CheckSizes(input_channels, input_height, input_width,
    output_channels, output_height, output_width);

  CHECK_LE(input_num, output_num);

  // For Blob use:
  const int height_offset = input_width;
  const int channel_offset = input_height * input_width;

  vector<const Dtype*> data_ptrs (1, NULL);
  for (int n = 0; n < input_num; ++n) {
    
    data_ptrs[0] = input_blob.cpu_data() + input_blob.offset(n);
    Dtype* transformed_data = transformed_blob->mutable_cpu_data()
    + transformed_blob->offset(n);

    UpdateState(input_channels, input_height, input_width);
    InternalTransform(data_ptrs, height_offset, channel_offset,
    output_channels, output_height, output_width, transformed_data);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Blob<Dtype>*>& input_blobs,
                              const vector<Blob<Dtype>*>& transformed_blobs) {
  CHECK_GT(input_blobs.size(), 0) << "There are no input Blobs";
  CHECK_EQ(input_blobs.size(), transformed_blobs.size()) <<
    "The size of input_blobs must be equal to the size of transformed_blobs";

  for (int item_id = 0; item_id < input_blobs.size(); ++item_id) {
    Transform(*(input_blobs[item_id]), transformed_blobs[item_id]);
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void DataTransformer<Dtype>::UpdateState(const int channels, const int height,
      const int width) {
  const int crop_size = param_.crop_size();
  CHECK_GE(height, crop_size);
  CHECK_GE(width, crop_size);
  // If not persistent or if persistent and reset then generate random
  // params if needed.
  if (!state_.persistent || state_.reset) {
    // If persistent only initialize the first time
    state_.input_channels = channels;
    state_.input_height = height;
    state_.input_width = width;
    if (state_.persistent) {
      state_.reset = false;
    }
    state_.do_mirror = param_.mirror() && Rand(2);
    if (crop_size) {
      if (phase_ == Caffe::TRAIN) {
        state_.h_off = Rand(height - crop_size + 1);
        state_.w_off = Rand(width - crop_size + 1);
      } else {
        state_.h_off = (height - crop_size) / 2;
        state_.w_off = (width - crop_size) / 2;
      }
    }
  }
  CHECK_EQ(channels, state_.input_channels)
    << "When persistent channels cannot change";
  CHECK_EQ(height, state_.input_height) << "When persistent height cannot change";
  CHECK_EQ(width, state_.input_width) << "When persistent width cannot change";
}

template<typename Dtype>
void DataTransformer<Dtype>::CheckSizes(const int input_channels,
    const int input_height, const int input_width, const int output_channels,
    const int output_height, const int output_width) {
  CHECK_EQ(output_channels, input_channels);
  CHECK_LE(output_height, input_height);
  CHECK_LE(output_width, input_width);
  if (param_.crop_size()) {
    CHECK_EQ(param_.crop_size(), output_height);
    CHECK_EQ(param_.crop_size(), output_width);
  } else {
    CHECK_EQ(input_height, output_height);
    CHECK_EQ(input_width, output_width);
  }
  if (param_.has_mean_file()) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
  }
  if (mean_values_.size() > 0) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1 && input_channels > 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < input_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template<typename Dtype, typename Datatype>
void DataTransformer<Dtype>::InternalTransform(
    const vector<const Datatype*> & data_ptrs, const int height_offset,
    const int channel_offset, const int output_height, const int output_width,
    const int output_channels, Dtype* transformed_data) {
  const int h_off = state_.h_off;
  const int w_off = state_.w_off;
  const bool do_mirror = state_.do_mirror;

  const int num_ptrs = data_ptrs.size();
  CHECK_GT(num_ptrs, 0);

  int top_index = 0;
  Datatype* data = data_ptrs[0];
  for (int h = 0; h < output_height; ++h) {
    if (num_ptrs > 1) {
      data = data_ptrs[h + h_off];
    }
    for (int w = 0; w < output_width; ++w) {
      for (int c = 0; c < output_channels; ++c) {
        if (do_mirror) {
          top_index = (c * output_height + h) * output_width + (output_width - 1 - w);
        } else {
          top_index = (c * output_height + h) * output_width + w;
        }
        int data_index =
          c * channel_offset + (h + h_off) * height_offset + w + w_off;
        Dtype pixel = static_cast<Dtype>(data[data_index]);
        if (has_mean_file) {
          transformed_data[top_index] =
            (pixel - data_mean_->data_at(0, c, h_off + h, w_off + w)) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] = (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
