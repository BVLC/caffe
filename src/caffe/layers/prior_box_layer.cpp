#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/prior_box_layer.hpp"

namespace caffe {

template <typename Dtype>
void PriorBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const PriorBoxParameter& prior_box_param =
      this->layer_param_.prior_box_param();
  CHECK_GT(prior_box_param.min_size_size(), 0) << "must provide min_size.";
  for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
    min_sizes_.push_back(prior_box_param.min_size(i));
    CHECK_GT(min_sizes_.back(), 0) << "min_size must be positive.";
  }
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.);
  flip_ = prior_box_param.flip();
  for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
    float ar = prior_box_param.aspect_ratio(i);
    bool already_exist = false;
    for (int j = 0; j < aspect_ratios_.size(); ++j) {
      if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_.push_back(ar);
      if (flip_) {
        aspect_ratios_.push_back(1./ar);
      }
    }
  }
  num_priors_ = aspect_ratios_.size() * min_sizes_.size();
  if (prior_box_param.max_size_size() > 0) {
    CHECK_EQ(prior_box_param.min_size_size(), prior_box_param.max_size_size());
    for (int i = 0; i < prior_box_param.max_size_size(); ++i) {
      max_sizes_.push_back(prior_box_param.max_size(i));
      CHECK_GT(max_sizes_[i], min_sizes_[i])
          << "max_size must be greater than min_size.";
      num_priors_ += 1;
    }
  }
  clip_ = prior_box_param.clip();
  if (prior_box_param.variance_size() > 1) {
    // Must and only provide 4 variance.
    CHECK_EQ(prior_box_param.variance_size(), 4);
    for (int i = 0; i < prior_box_param.variance_size(); ++i) {
      CHECK_GT(prior_box_param.variance(i), 0);
      variance_.push_back(prior_box_param.variance(i));
    }
  } else if (prior_box_param.variance_size() == 1) {
    CHECK_GT(prior_box_param.variance(0), 0);
    variance_.push_back(prior_box_param.variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }

  if (prior_box_param.has_img_h() || prior_box_param.has_img_w()) {
    CHECK(!prior_box_param.has_img_size())
        << "Either img_size or img_h/img_w should be specified; not both.";
    img_h_ = prior_box_param.img_h();
    CHECK_GT(img_h_, 0) << "img_h should be larger than 0.";
    img_w_ = prior_box_param.img_w();
    CHECK_GT(img_w_, 0) << "img_w should be larger than 0.";
  } else if (prior_box_param.has_img_size()) {
    const int img_size = prior_box_param.img_size();
    CHECK_GT(img_size, 0) << "img_size should be larger than 0.";
    img_h_ = img_size;
    img_w_ = img_size;
  } else {
    img_h_ = 0;
    img_w_ = 0;
  }

  if (prior_box_param.has_step_h() || prior_box_param.has_step_w()) {
    CHECK(!prior_box_param.has_step())
        << "Either step or step_h/step_w should be specified; not both.";
    step_h_ = prior_box_param.step_h();
    CHECK_GT(step_h_, 0.) << "step_h should be larger than 0.";
    step_w_ = prior_box_param.step_w();
    CHECK_GT(step_w_, 0.) << "step_w should be larger than 0.";
  } else if (prior_box_param.has_step()) {
    const float step = prior_box_param.step();
    CHECK_GT(step, 0) << "step should be larger than 0.";
    step_h_ = step;
    step_w_ = step;
  } else {
    step_h_ = 0;
    step_w_ = 0;
  }

  offset_ = prior_box_param.offset();
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);
  // Since all images in a batch has same height and width, we only need to
  // generate one set of priors which can be shared across all images.
  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  top_shape[2] = layer_width * layer_height * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = bottom[1]->width();
    img_height = bottom[1]->height();
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim = layer_height * layer_width * num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset_) * step_w;
      float center_y = (h + offset_) * step_h;
      float box_width, box_height;
      for (int s = 0; s < min_sizes_.size(); ++s) {
        int min_size_ = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size_;
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;

        if (max_sizes_.size() > 0) {
          CHECK_EQ(min_sizes_.size(), max_sizes_.size());
          int max_size_ = max_sizes_[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_size_ * max_size_);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        // rest of priors
        for (int r = 0; r < aspect_ratios_.size(); ++r) {
          float ar = aspect_ratios_[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }
  // set the variance.
  top_data += top[0]->offset(0, 1);
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PriorBoxLayer);
REGISTER_LAYER_CLASS(PriorBox);

}  // namespace caffe
