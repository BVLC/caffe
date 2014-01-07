// Copyright 2013 Ross Girshick

#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'cropsize' indicates the desired warped size

// TODO(rbg):
//  - try uniform sampling over classes

namespace caffe {

template <typename Dtype>
void* WindowDataLayerPrefetch(void* layer_pointer) {
  WindowDataLayer<Dtype>* layer = 
      reinterpret_cast<WindowDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const bool mirror = layer->layer_param_.mirror();
  const float fg_fraction = layer->layer_param_.det_fg_fraction();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_off = (layer->data_mean_.width() - cropsize) / 2;
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  cv::Size cv_crop_size(cropsize, cropsize);

//  CHECK_EQ(mean_width, mean_height);
//  CHECK_EQ(mean_width, 256);
//  CHECK_EQ(mean_off, 14);

  const int num_fg = static_cast<int>(static_cast<float>(batchsize) 
      * fg_fraction);
  const int num_samples[2] = { batchsize - num_fg, num_fg };

  int itemid = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      vector<float> window = (is_fg) 
          ? layer->fg_windows_[rand() % layer->fg_windows_.size()]
          : layer->bg_windows_[rand() % layer->bg_windows_.size()];

      // load the image containing the window
      std::pair<std::string, vector<int> > image = 
          layer->image_database_[window[WindowDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return (void*)NULL;
      }
      const int channels = cv_img.channels();
//      CHECK_EQ(channels, 3);

      // crop window out of image and warp it
      const int x1 = window[WindowDataLayer<Dtype>::X1];
      const int y1 = window[WindowDataLayer<Dtype>::Y1];
      const int x2 = window[WindowDataLayer<Dtype>::X2];
      const int y2 = window[WindowDataLayer<Dtype>::Y2];
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img, 
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
      
      // horizontal flip at random
//      bool is_mirror = false;
      if (mirror && rand() % 2) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
//        is_mirror = true;
      }
      
      // TODO(rbg): this could probably be made more efficient
      // but this thread finishes before the GPU is ready, 
      // so it's fine for now
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize; ++h) {
          for (int w = 0; w < cropsize; ++w) {
            Dtype pixel = 
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

            top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
                = (pixel
                    - mean[(c * mean_height + h + mean_off) 
                           * mean_width + w + mean_off])
                  * scale;
          }
        }
      }

      // get window label
      top_label[itemid] = window[WindowDataLayer<Dtype>::LABEL];

//      string file_id;
//      std::stringstream ss;
//      ss << rand();
//      ss >> file_id;
//      std::ofstream inf((string("dump/") + file_id + string("_info.txt")).c_str(), std::ofstream::out);
//      inf << image.first << std::endl 
//          << x1+1 << std::endl
//          << y1+1 << std::endl
//          << x2+1 << std::endl
//          << y2+1 << std::endl
//          << is_mirror << std::endl
//          << top_label[itemid] << std::endl
//          << is_fg << std::endl;
////          << "is_fg: " << is_fg << std::endl
////          << "label: " << top_label[itemid] << " " << window[WindowDataLayer<Dtype>::LABEL] << std::endl
////          << "num bg samples: " << num_samples[0] << std::endl
////          << "num fg samples: " << num_samples[1];
//      inf.close();
//      std::ofstream top_data_file((string("dump/") + file_id + string("_data.txt")).c_str(), 
//            std::ofstream::out | std::ofstream::binary);
//      for (int c = 0; c < channels; ++c) {
//        for (int h = 0; h < cropsize; ++h) {
//          for (int w = 0; w < cropsize; ++w) {
//            top_data_file.write(
//                reinterpret_cast<char*>(&top_data[((itemid * channels + c) 
//                                                   * cropsize + h) * cropsize + w]),
//                sizeof(Dtype));
//          }
//        }
//      }
//      top_data_file.close();

      itemid++;
    }
  }

  return (void*)NULL;
}


template <typename Dtype>
void WindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures 
  // that hold windows: one for foreground (object) windows and one 
  // for background (non-object) windows. We use an overlap threshold 
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Window data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Window data Layer prodcues two blobs as output.";

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: " 
      << this->layer_param_.det_fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: " 
      << this->layer_param_.det_bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.det_fg_fraction();

  std::ifstream infile(this->layer_param_.source().c_str());
  CHECK(infile.good()) << "Failed to open window file " 
      << this->layer_param_.source() << std::endl;

  vector<float> label_hist(21);
  std::fill(label_hist.begin(), label_hist.end(), 0);

  string hashtag;
  int image_index, channels;
  while (infile >> hashtag >> image_index) {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    // read each box
    int num_windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;

      vector<float> window(WindowDataLayer::NUM);
      window[WindowDataLayer::IMAGE_INDEX] = image_index;
      window[WindowDataLayer::LABEL] = label;
      window[WindowDataLayer::OVERLAP] = overlap;
      window[WindowDataLayer::X1] = x1;
      window[WindowDataLayer::Y1] = y1;
      window[WindowDataLayer::X2] = x2;
      window[WindowDataLayer::Y2] = y2;
      
      // add window to foreground list or background list
      if (overlap >= this->layer_param_.det_fg_threshold()) {
        CHECK_GT(window[WindowDataLayer::LABEL], 0);
        fg_windows_.push_back(window);
      } else if (overlap < this->layer_param_.det_bg_threshold()) {
        // background window, force label and overlap to 0
        window[WindowDataLayer::LABEL] = 0;
        window[WindowDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
      }
      label_hist[window[WindowDataLayer::LABEL]]++;
    }

    if (image_index % 1 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " " 
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  }

  LOG(INFO) << "Number of images: " << image_index;

  for (int i = 0; i < 21; ++i) {
    LOG(INFO) << "class " << i << " has " << label_hist[i] << " samples";
  }

  // image
  int cropsize = this->layer_param_.cropsize();
  CHECK_GT(cropsize, 0);
  (*top)[0]->Reshape(
      this->layer_param_.batchsize(), channels, cropsize, cropsize);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.batchsize(), channels, cropsize, cropsize));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), 1, 1, 1));

  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, cropsize, cropsize);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, WindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void WindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, WindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void WindowDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, WindowDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype WindowDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype WindowDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(WindowDataLayer);

}  // namespace caffe
