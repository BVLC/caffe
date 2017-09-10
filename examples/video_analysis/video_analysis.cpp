#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <math.h>
#include "caffe/data_transformer.hpp"
#include <caffe/caffe.hpp>

#if defined(USE_OPENCV)
using namespace caffe;  // NOLINT(build/namespaces)

struct detect_result {
  int imgid;
  float classid;
  float confidence;
  float left;
  float right;
  float top;
  float bottom;
};

enum ColorFormat {
  VA_RGB = 0,
  VA_BGR = 1
};

template <typename Dtype>
class Detector {
public:
  Detector(const string& model_file,
           const string& weights_file,
           int gpu,
           int batch_size);
  void Preprocess(const vector<cv::Mat> &imgs);

  void Detect(vector<vector<detect_result>> &result,
              int iter_count,
              bool visualize = false,
              bool step_mode = false);

  void ShowResult(const vector<cv::Mat> &imgs,
                  const vector<vector<detect_result>> objects,
                  bool step_mode);
  ~Detector() {
    for (int i = 0; i < input_blobs_.size(); i++)
      delete input_blobs_[i];
    delete data_transformer_;
  }

private:

  shared_ptr<Net<Dtype> > net_;
  cv::Size input_blob_size_;
  cv::Size image_size_;
  int num_channels_;
  int batch_size_;
  bool use_yolo_format_;
  vector<Blob<Dtype>*> input_blobs_;
  const vector<cv::Mat> *origin_imgs_;
  DataTransformer<Dtype> *data_transformer_;
  ColorFormat input_color_format_;
};

template <typename Dtype>
Detector<Dtype>::Detector(const string& model_file,
                   const string& weights_file,
                   int gpu,
                   int batch_size) {
    // Set device id and mode
  if (gpu != -1) {
    std::cout << "Use GPU with device ID " << gpu << std::endl;
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu);
  }
  else {
    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);
  }
  batch_size_ = batch_size;
  /* Load the network. */
  net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_blob_size_ = cv::Size(input_layer->width(), input_layer->height());

  // Check whether the model we will using.
  // Different models need different preprocessing parameters.
  const shared_ptr<Layer<Dtype>> output_layer = net_->layers().back();
  TransformationParameter transform_param;
  caffe::ResizeParameter *resize_param = transform_param.mutable_resize_param();
  if (output_layer->layer_param().type() == "YoloDetectionOutput") {
    use_yolo_format_ = !output_layer->layer_param().
                        yolo_detection_output_param().ssd_format();
    resize_param->set_resize_mode(caffe::ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
    resize_param->add_pad_value(127.5);
    transform_param.set_scale(1./255.);
    transform_param.set_force_color(true);
    std::cout << "Using Yolo: " << net_->name() << std::endl;
    input_color_format_ = VA_RGB;
  } else if (output_layer->layer_param().type() == "DetectionOutput") {
    use_yolo_format_ = false;
    resize_param->set_resize_mode(caffe::ResizeParameter_Resize_mode_WARP);
    if (net_->name().find("MobileNet") != std::string::npos) {
      transform_param.add_mean_value(127.5);
      transform_param.add_mean_value(127.5);
      transform_param.add_mean_value(127.5);
      transform_param.set_scale(1./127.5);
      std::cout << "Using SSD(MobileNet)." << std::endl;
    } else {
      // For standard SSD VGG or DSOD
      transform_param.add_mean_value(104);
      transform_param.add_mean_value(117);
      transform_param.add_mean_value(123);
      std::cout << "Using SSD : " << net_->name() << std::endl;
    }
    input_color_format_ = VA_BGR;
  } else {
    std::cerr << "The model is not a valid object detection model."
              << std::endl;
    exit(-1);
  }
  resize_param->set_width(input_blob_size_.width);
  resize_param->set_height(input_blob_size_.height);
  resize_param->set_prob(1.0);
  resize_param->add_interp_mode(caffe::ResizeParameter_Interp_mode_LINEAR);
  data_transformer_ = new DataTransformer<Dtype>(transform_param,
                                                 TEST,
                                                 Caffe::GetDefaultDevice());
}

static void
FixupChannels(vector <cv::Mat> &imgs, int num_channels,
              enum ColorFormat color_format) {
  for (int i = 0; i < imgs.size(); i++) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat img = imgs[i];
    if (img.channels() != num_channels) {
      cv::Mat sample;
      if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
      else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample,
                     color_format == VA_BGR ?
                     cv::COLOR_BGRA2BGR : cv::COLOR_BGRA2RGB);
      else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample,
                     color_format == VA_BGR ?
                     cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2RGB);
      else {
        // Should not enter here, just in case.
        if (color_format == VA_BGR)
          sample = img;
        else
          cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
      }
      imgs[i] = sample;
    }
  }
}

template <typename Dtype>
void Detector<Dtype>::Preprocess(const vector<cv::Mat> &imgs) {

  if (imgs.size() == 0)
    return;
  origin_imgs_ = & imgs;
  int batch_id = 0;

  image_size_.width = imgs[0].cols;
  image_size_.height = imgs[0].rows;
  int batch_count = imgs.size() / batch_size_;

  for (batch_id = 0; batch_id < batch_count; batch_id++) {
    Blob<Dtype> * blob = new Blob<Dtype>;
    blob->Reshape(batch_size_,
                  num_channels_,
                  input_blob_size_.height,
                  input_blob_size_.width);
    int pos = batch_id * batch_size_;
    vector<cv::Mat> batch_imgs(imgs.begin() + pos,
                               imgs.begin() + pos + batch_size_);
    FixupChannels(batch_imgs, num_channels_, input_color_format_);
    data_transformer_->Transform(batch_imgs, blob);
    input_blobs_.push_back(blob);
  }

  if (batch_id * batch_size_ < imgs.size()) {
    Blob<Dtype> * blob = new Blob<Dtype>;
    blob->Reshape(imgs.size() - batch_id * batch_size_,
                  num_channels_,
                  input_blob_size_.height,
                  input_blob_size_.width);
    int pos = batch_id * batch_size_;
    int batch_size = imgs.size() - batch_id * batch_size_;
    const vector<cv::Mat> batch_imgs(imgs.begin() + pos,
                                     imgs.begin() + pos + batch_size);
    input_blobs_.push_back(blob);
  }
}

template <typename Dtype>
void Detector<Dtype>::ShowResult(const vector<cv::Mat> &imgs,
                                 const vector<vector<detect_result>> objects,
                                 bool step_mode)
{
    for (int i = 0; i < objects.size(); i++) {
      if (objects[i].size() == 0)
        continue;
      int frame_num = objects[i][0].imgid;
      cv::Mat img = imgs[frame_num].clone();
      for (int j = 0; j < objects[i].size(); j++) {
        detect_result obj = objects[i][j];
	cv::rectangle(img,
                      cvPoint(obj.left, obj.top),
                      cvPoint(obj.right, obj.bottom),
                      cv::Scalar(255, 242, 35));
	std::stringstream ss;
	ss << obj.classid << "/" << obj.confidence;
	std::string  text = ss.str();
	cv::putText(img,
                    text,
                    cvPoint(obj.left, obj.top + 20),
                    cv::FONT_HERSHEY_PLAIN,
                    1.0f,
                    cv::Scalar(0, 255, 255));
      }
      cv::imshow("detections", img);
      int wait_ms = step_mode ? 0 : 1;
      int key = cv::waitKey(static_cast<char>(wait_ms));
      if (key == 'q')
        exit(1);
    }
}

// Normalized coordinate fixup for yolo
float fixup_norm_coord(float coord, float ratio) {
  if (ratio >= 1)
    return coord;
  else
    return (coord - (1. - ratio)/2) / ratio;
}

template <typename Dtype>
void Detector<Dtype>::Detect(vector<vector<detect_result>> &all_objects,
                             int iter_count,
                             bool visualize,
                             bool step_mode) {
  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  input_layer->ReshapeLike(*input_blobs_[0]);
  net_->Reshape();

  int batch_to_detect = iter_count * input_blobs_.size();
  // iter_count 0 is for warm up to handle only 1 batch.
  if (batch_to_detect == 0)
    batch_to_detect = 1;
  int w = image_size_.width;
  int h = image_size_.height;

  for (int batch_id = 0; batch_id < batch_to_detect; batch_id++) {
    int real_batch_id = batch_id % input_blobs_.size();
    int batch_size = input_blobs_[real_batch_id]->num();
    input_layer->ReshapeLike(*input_blobs_[real_batch_id]);
    input_layer->ShareData(*input_blobs_[real_batch_id]);
    net_->Forward();
    Blob<Dtype>* result_blob = net_->output_blobs()[0];
    const Dtype* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<detect_result>> objects(batch_size);
    for (int k = 0; k < num_det * 7; k += 7) {
      detect_result object;
      object.imgid = (int)result[k + 0] + real_batch_id * batch_size;
      object.classid = (int)result[k + 1];
      object.confidence = result[k + 2];
      if (use_yolo_format_) {
        object.left = (int)(fixup_norm_coord((result[k + 3] -
                         result[k + 5] / 2.0), float(w) / h) * w);
        object.right = (int)(fixup_norm_coord((result[k + 3] +
                         result[k + 5] / 2.0), float(w) / h) * w);
        object.top = (int)(fixup_norm_coord((result[k + 4] -
                        result[k + 6] / 2.0), float(h) / w) * h);
        object.bottom = (int)(fixup_norm_coord((result[k + 4] +
                           result[k + 6] / 2.0), float(h) / w) * h);
      } else {
        object.left = (int)(result[k + 3] * w);
        object.top = (int)(result[k + 4] * h);
        object.right = (int)(result[k + 5] * w);
        object.bottom = (int)(result[k + 6] * h);
      }
      if (object.left < 0) object.left = 0;
      if (object.top < 0) object.top = 0;
      if (object.right >= w) object.right = w - 1;
      if (object.bottom >= h) object.bottom = h - 1;
      objects[result[k + 0]].push_back(object);
    }
    if (visualize)
      ShowResult(*origin_imgs_, objects, step_mode);
    all_objects.insert(all_objects.end(), objects.begin(), objects.end());
  }
}

int main(int argc, char** argv) {

#if CV_MAJOR_VERSION >= 3
  const char* keys =
    "{ model model_file      | <none> | model file }"
    "{ weights weights_file  | <none> | weights file }"
    //"{ img image             | <none> | path to image }"
    "{ video                 | <none> | path to video }"
    //"{ out out_file          |        | output image file }"
    "{ s step                | false  | true for step mode }"
    "{ i iter                | 1      | iterations to be run }"
    "{ v visualize           | false   | visualize output }"
    "{ g gpu                 | 0      | gpu device }"
    "{ c cpu                 | false  | use cpu device }"
    "{ fp16 use_fp16         | false  | use fp16 forward engine. }"
    "{ bs batch_size         | 1      | batch size }"
    "{ frame_count f         | 1024   | process frame count in the video}"
    "{ help                  | false  | display this help and exit  }"
    ;
#else
  const char* keys =
    "{ model   | model_file      | <none> | model file }"
    "{ weights | weights_file    | <none> | weights file }"
    //"{ img     | image           | <none> | path to image }"
    "{ video   | video           | <none> | path to video }"
    //"{ out     | out_file        |        | output image file }"
    "{ s       | step            | false  | true for step mode }"
    "{ i       | iter            | 1      | iterations to be run }"
    "{ v       | visualize       | false   | visualize output }"
    "{ g       | gpu             | 0     | gpu device }"
    "{ c       | cpu             | false | use cpu device }"
    "{ fp16    | use_fp16        | false  | use fp16 forward engine. }"
    "{ bs      | batch_size      | 1      | batch size }"
    "{ f       | frame_count     | 1024   | process frame count in the video}"
    "{ h       | help            | false  | display this help and exit  }"
    ;
#endif

  cv::CommandLineParser parser(argc, argv, keys);
  const string model_file = parser.get<std::string>("model_file");
  const string weights_file = parser.get<std::string>("weights_file");
  const string video_file = parser.get<std::string>("video");
  int iter = parser.get<int>("iter");
  bool visualize = parser.get<bool>("visualize");
  int gpu = parser.get<int>("gpu");
  bool cpu = parser.get<bool>("cpu");
  int batch_size = parser.get<int>("batch_size");
  bool use_fp16 = parser.get<bool>("use_fp16");
  bool step_mode = parser.get<bool>("step");
  int max_frame_count = parser.get<int>("frame_count");

  if (model_file == "" ||
      weights_file == "" ||
      video_file == "") {
#if CV_MAJOR_VERSION >= 3
    parser.printMessage();
#else
    parser.printParams();
#endif
    exit(-1);
  }

  if (cpu)
    gpu = -1;
  std::streambuf* buf = std::cout.rdbuf();
  std::ostream out(buf);
  cv::VideoCapture cap_;
  int total_frames = 0;
  cv::Mat tmpMat;

  if (!cap_.open(video_file))
    LOG(FATAL) << "Failed to open video: " << video_file;

  total_frames = cap_.get(CV_CAP_PROP_FRAME_COUNT);
  total_frames = (total_frames > max_frame_count) ?
                  max_frame_count : total_frames;
  // For benchmark mode, we limit the frames to 8
  if (!visualize && total_frames > 8)
    total_frames = 8;
  std::cout << "\nTotal frame number = " << total_frames;
  vector<cv::Mat> imgCache(total_frames);
  for (int i = 0; i < total_frames; i++) {
    cap_ >> imgCache[i];
  }

  vector<vector<struct detect_result>> objects;
  double warm_up_time = 0;
  double detect_time = 0;
  // Initialize the network.
  if (use_fp16) {
#ifdef HAS_HALF_SUPPORT
    Detector<half> detector(model_file, weights_file, gpu, batch_size);
    Timer warm_up_timer, detect_timer;
    detector.Preprocess(imgCache);
    // Detect one batch to warm up.
    warm_up_timer.Start();
    detector.Detect(objects, 0);
    detector.Detect(objects, 0);
    warm_up_timer.Stop();
    objects.clear();
    warm_up_time = warm_up_timer.MilliSeconds();

    detect_timer.Start();
    detector.Detect(objects, iter, visualize, step_mode);
    detect_timer.Stop();
    detect_time = detect_timer.MilliSeconds();
#else
    std::cout << "fp16 is not supported." << std::endl;
#endif
  } else {
    Detector<float> detector(model_file, weights_file, gpu, batch_size);
    Timer warm_up_timer, detect_timer;
    detector.Preprocess(imgCache);
    // Detect one batch to warm up.
    warm_up_timer.Start();
    detector.Detect(objects, 0);
    warm_up_timer.Stop();
    objects.clear();
    warm_up_time = warm_up_timer.MilliSeconds();
    detect_timer.Start();
    detector.Detect(objects, iter, visualize, step_mode);
    detect_timer.Stop();
    detect_time = detect_timer.MilliSeconds();
  }

  std::cout << "Warm up time is "
            << warm_up_time
            << "ms." << std::endl;

  std::cout << "Detect time is "
            << detect_time
            << "ms." << std::endl;

  double fps = total_frames * iter / (detect_time / 1000.);
  std::cout << "Average frames per second is " << fps << "." << std::endl
            << "Average latency is " << 1000./fps << " ms." << std::endl;
  return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV and half floating point support."
        << "compile with USE_OPENCV and USE_ISAAC.";
}
#endif  // USE_OPENCV

