#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"



boost::mt19937 gen;
const double prob_eps = 0.01;

int roll_weighted_die(const std::vector<double> probabilities) {
  std::vector<double> cumulative;
  std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
      std::back_inserter(cumulative));
  boost::uniform_real<> dist(0, cumulative.back());
  boost::variate_generator<boost::mt19937&,
      boost::uniform_real<> > die(gen, dist);

  // Find the position within the sequence and add 1
  return (std::lower_bound(cumulative.begin(), cumulative.end(), die())
      - cumulative.begin());
}

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
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

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
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
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

  template<typename Dtype>
  void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
      Blob<Dtype>* transformed_blob) {
    const int img_channels = cv_img.channels();
    int img_height = cv_img.rows;
    int img_width = cv_img.cols;

    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, img_channels);
    CHECK_GE(num, 1);

    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

    const int crop_size = param_.crop_size();
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool do_vert_mirror = param_.vertical_mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    const bool has_mean_values = mean_values_.size() > 0;
    const int rotation_count = (param_.random_90_deg_rot()) ? Rand(4) : 0;
    cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;

    const int num_resize_policies = param_.resize_param_size();
    const int num_noise_policies = param_.noise_param_size();

    if (num_resize_policies > 0) {
      std::vector<double> probabilities;
      double prob_sum = 0;
      for (int i = 0; i < num_resize_policies; i++) {
        const double prob = param_.resize_param(i).prob();
        CHECK_GE(prob, 0);
        CHECK_LE(prob, 1);
        prob_sum+=prob;
        probabilities.push_back(prob);
      }
      CHECK_NEAR(prob_sum, 1.0, prob_eps);
      int policy_num = roll_weighted_die(probabilities);
      cv_resized_image = ApplyResize(cv_img, param_.resize_param(policy_num));
    } else {
      cv_resized_image = cv_img;
    }

    if (num_noise_policies > 0) {
      std::vector<double> probabilities;
      double prob_sum = 0;
      for (unsigned int i = 0; i < num_noise_policies; i++) {
        const double prob = param_.noise_param(i).prob();
        CHECK_GE(prob, 0);
        CHECK_LE(prob, 1);
        prob_sum+=prob;
        probabilities.push_back(prob);
      }
      CHECK_NEAR(prob_sum, 1.0, prob_eps);
      int policy_num = roll_weighted_die(probabilities);
      cv_noised_image = ApplyNoise(cv_resized_image,
          param_.noise_param(policy_num));

    } else {
      cv_noised_image = cv_resized_image;
    }

    CHECK_GT(img_channels, 0);

    Dtype* mean = NULL;
    if (has_mean_file) {
      CHECK_EQ(img_channels, data_mean_.channels());
      mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
      CHECK(mean_values_.size() == 1 ||
          mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
      if (img_channels > 1 && mean_values_.size() == 1) {
        // Replicate the mean_value for simplicity
        for (int c = 1; c < img_channels; ++c) {
          mean_values_.push_back(mean_values_[0]);
        }
      }
    }

    CHECK_GE(cv_noised_image.rows, crop_size);
    CHECK_GE(cv_noised_image.cols, crop_size);
    img_height = cv_noised_image.rows;
    img_width = cv_noised_image.cols;

    int h_off = 0;
    int w_off = 0;
    if (crop_size) {
      CHECK_EQ(crop_size, height);
      CHECK_EQ(crop_size, width);
      // We only do random crop when we do training.
      if (phase_ == TRAIN) {
        h_off = Rand(img_height - crop_size + 1);
        w_off = Rand(img_width - crop_size + 1);
      } else {
        h_off = (img_height - crop_size) / 2;
        w_off = (img_width - crop_size) / 2;
      }
      cv::Rect roi(w_off, h_off, crop_size, crop_size);
      cv_cropped_image = cv_noised_image(roi);
    } else {
      cv_cropped_image = cv_noised_image;
    }

    CHECK_EQ(cv_cropped_image.rows, height);
    CHECK_EQ(cv_cropped_image.cols, width);
    if (has_mean_file) {
      CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
      CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
    }
    CHECK(cv_cropped_image.data);

    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    int top_index;
    for (int h = 0; h < height; ++h) {
      const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
      int img_index = 0;
      int h_idx = h;
      if (do_vert_mirror) {
        h_idx = height - 1 - h;
      }
      for (int w = 0; w < width; ++w) {
        int w_idx = w;
        if (do_mirror) {
          w_idx = (width - 1 - w);
        }
        int h_idx_real = h_idx;
        int w_idx_real = w_idx;
        if (rotation_count == 1) {
          int temp = w_idx_real;
          w_idx_real = height - 1 - h_idx_real;
          h_idx_real = temp;
        } else if (rotation_count == 2) {
          w_idx_real = width - 1 - w_idx_real;
          h_idx_real = height - 1 - h_idx_real;
        } else if (rotation_count == 3) {
          int temp = h_idx_real;
          h_idx_real = width - 1 - w_idx_real;
          w_idx_real = temp;
        }
        for (int c = 0; c < img_channels; ++c) {
          top_index = (c * height + h_idx_real) * width + w_idx_real;
          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          if (has_mean_file) {
            int mean_index = (c * img_height + h_off + h_idx_real) * img_width
                + w_off + w_idx_real;
            transformed_data[top_index] =
                (pixel - mean[mean_index]) * scale;
          } else {
            if (has_mean_values) {
              transformed_data[top_index] =
                  (pixel - mean_values_[c]) * scale;
            } else {
              transformed_data[top_index] = pixel * scale;
            }
          }
        }
      }
    }
  }
template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
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

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
