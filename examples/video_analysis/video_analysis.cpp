#include <caffe/caffe.hpp>
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

#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
using namespace caffe;  // NOLINT(build/namespaces)
#define MAX_FRAMES 1024

struct detect_result {
  int imgid;
  float classid;
  float confidence;
  float left;
  float right;
  float top;
  float bottom;
};

enum DetectionModel {
  SSD = 1,
  YOLO = 2
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
              bool visualize = false);

  void ShowResult(const vector<cv::Mat> &imgs,
                     const vector<vector<detect_result>> objects);
  ~Detector() {
    for (int i = 0; i < input_blobs_.size(); i++)
      delete input_blobs_[i];
  }

private:
  void Preprocess(const cv::Mat& img,
                  std::vector<Dtype *> input_channels);
  void WrapInputLayer(std::vector<Dtype *> &input_channels,
                      Blob<Dtype>* blob,
                      int batch_id);

  shared_ptr<Net<Dtype> > net_;
  cv::Size input_blob_size_;
  cv::Size image_size_;
  int num_channels_;
  int batch_size_;
  bool use_yolo_format_;
  DetectionModel model_;
  vector<Blob<Dtype>*> input_blobs_;
  const vector<cv::Mat> * origin_imgs_;
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

  // Check whether we are SSD or yolo.
  const shared_ptr<Layer<Dtype>> output_layer = net_->layers().back();
  if (output_layer->layer_param().type() == "YoloDetectionOutput") {
    use_yolo_format_ = !output_layer->layer_param().
                        yolo_detection_output_param().ssd_format();
    model_ = YOLO;
  } else if (output_layer->layer_param().type() == "DetectionOutput") {
    model_ = SSD;
    use_yolo_format_ = false;
  } else {
    std::cerr << "The model is not a valid object detection model."
              << std::endl;
    exit(-1);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
template <typename Dtype>
void Detector<Dtype>::WrapInputLayer(std::vector<Dtype *> &input_channels,
                                     Blob<Dtype>* blob,
                                     int batch_id) {
  int width = blob->width();
  int height = blob->height();
  Dtype* input_data = blob->mutable_cpu_data();
  input_data += batch_id * width * height * blob->channels();
  for (int i = 0; i < blob->channels(); ++i) {
    input_channels.push_back(input_data);
    input_data += width * height;
  }
}

template <typename Dtype>
void Detector<Dtype>::Preprocess(const cv::Mat& img,
  std::vector<Dtype *> input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample,
                 model_ == YOLO ?
                 cv::COLOR_BGRA2RGB : cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample,
                 model_ == YOLO ?
                 cv::COLOR_GRAY2RGB : cv::COLOR_GRAY2BGR);
  else
    sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_blob_size_)
    cv::resize(sample, sample_resized, input_blob_size_);
  else
    sample_resized = sample;
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);
  cv::Mat sample_normalized;
  if (model_ == YOLO)
    cv::divide(sample_float, 255.0, sample_normalized);
  else
    sample_normalized = sample_float;
  for (int_tp i = 0; i < input_blob_size_.height; i++) {
    for (int_tp j = 0; j < input_blob_size_.width; j++) {
      int pos = i * input_blob_size_.width + j;
      if (num_channels_ == 3) {
        cv::Vec3f pixel = sample_normalized.at<cv::Vec3f>(i, j);
        input_channels[0][pos] = pixel.val[0];
        input_channels[1][pos] = pixel.val[1];
        input_channels[2][pos] = pixel.val[2];
      }
      else {
        cv::Scalar pixel = sample_normalized.at<float>(i, j);
        input_channels[0][pos] = pixel.val[0];
      }
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
    for (int j = 0; j < batch_size_; j++) {
      assert(image_size_.width == imgs[batch_id * batch_size_ + j].cols);
      assert(image_size_.height == imgs[batch_id * batch_size_ + j].rows);
      std::vector<Dtype *> input_channels;
      WrapInputLayer(input_channels, blob, j);
      Preprocess(imgs[batch_id * batch_size_ + j], input_channels);
    }
    input_blobs_.push_back(blob);
  }

  if (batch_id * batch_size_ < imgs.size()) {
    Blob<Dtype> * blob = new Blob<Dtype>;
    blob->Reshape(imgs.size() - batch_id * batch_size_,
                  num_channels_,
                  input_blob_size_.height,
                  input_blob_size_.width);
    for (int i = batch_id * batch_size_, j = 0; i < imgs.size(); j++, i++) {
      assert(image_size_.width == imgs[batch_id * batch_size_ + j].cols);
      assert(image_size_.height == imgs[batch_id * batch_size_ + j].rows);
      vector<Dtype *> input_channels;
      WrapInputLayer(input_channels, blob, j);
      Preprocess(imgs[batch_id * batch_size_ + j], input_channels);
    }
    input_blobs_.push_back(blob);
  }
}

template <typename Dtype>
void Detector<Dtype>::ShowResult(const vector<cv::Mat> &imgs,
                                    const vector<vector<detect_result>> objects)
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
      if (cv::waitKey(static_cast<char>(1)) == 27) {
        exit(0);
      }
    }
}

template <typename Dtype>
void Detector<Dtype>::Detect(vector<vector<detect_result>> &all_objects,
                             int iter_count,
                             bool visualize) {
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
        object.left = (int)((result[k + 3] - result[k + 5] / 2.0) * w);
        object.right = (int)((result[k + 3] + result[k + 5] / 2.0) * w);
        object.top = (int)((result[k + 4] - result[k + 6] / 2.0) * h);
        object.bottom = (int)((result[k + 4] + result[k + 6] / 2.0) * h);
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
      ShowResult(*origin_imgs_, objects);
    all_objects.insert(all_objects.end(), objects.begin(), objects.end());
  }
}

int main(int argc, char** argv) {

  if (argc < 3) {
    return 1;
  }

#if CV_MAJOR_VERSION >= 3
  const char* keys =
    "{ model model_file      | <none> | model file            }"
    "{ weights weights_file  | <none> | weights file            }"
    "{ img image             | <none> | path to image }"
    "{ video                 | <none> | path to video }"
    "{ out out_file          |        | output image file          }"
    "{ f fps                 | 1200   | target fps (0 for unlimited) }"
    "{ i iter                | 1      | iterations to be run }"
    "{ v visualize           | true   | visualize output }"
    "{ g gpu                 | 0      | gpu device }"
    "{ c cpu                 | false  | use cpu device }"
    "{ fp16 use_fp16         | false  | use fp16 forward engine. }"
    "{ bs batch_size         | 1      | batch size }"
    "{ help                  | false  | display this help and exit  }"
    ;
#else
  const char* keys =
    "{ model   | model_file      | <none> | model file            }"
    "{ weights | weights_file    | <none> | weights file            }"
    "{ img     | image           | <none> | path to image }"
    "{ video   | video           | <none> | path to video }"
    "{ out     | out_file        |        | output image file          }"
    "{ f       | fps             | 1200   | target fps (0 for unlimited) }"
    "{ i       | iter            | 1      | iterations to be run }"
    "{ v       | visualize       | true   | visualize output }"
    "{ g       | gpu             | 0     | gpu device }"
    "{ c       | cpu             | false | use cpu device }"
    "{ fp16    | use_fp16        | false  | use fp16 forward engine. }"
    "{ bs      | batch_size      | 1      | batch size }"
    "{ h       | help            | false  | display this help and exit  }"
    ;
#endif

  cv::CommandLineParser parser(argc, argv, keys);

  const string model_file = parser.get<std::string>("model_file");
  const string weights_file = parser.get<std::string>("weights_file");
  const string video_file = parser.get<std::string>("video");
  const string image_file = parser.get<std::string>("image");
  const string out_file = parser.get<std::string>("out_file");
  int iter = parser.get<int>("iter");
  int targetFPS = parser.get<int>("fps");
  bool visualize = parser.get<bool>("visualize");
  int gpu = parser.get<int>("gpu");
  bool cpu = parser.get<bool>("cpu");
  int batch_size = parser.get<int>("batch_size");
  bool use_fp16 = parser.get<bool>("use_fp16");

  if (cpu)
    gpu = -1;
  std::streambuf* buf = std::cout.rdbuf();
  std::ostream out(buf);
  bool loadVideo = false;
  vector<cv::Mat> imgCache;
  cv::VideoCapture cap_;
  int total_frames = 0;
  double delayMilliSeconds = targetFPS == 0 ? 1.0 : 1000.0 / targetFPS;
  std::cout << "\ndelayMillisSeconds value = " << delayMilliSeconds << "\n";
  cv::Mat tmpMat;
  if (video_file != "")
  {
      loadVideo = true;
      if (!cap_.open(video_file))
      {
          LOG(FATAL) << "Failed to open video: " << video_file;
      }
      total_frames = cap_.get(CV_CAP_PROP_FRAME_COUNT);
      total_frames = (total_frames > MAX_FRAMES) ? MAX_FRAMES : total_frames;
      std::cout << "\nTotal frame number = " << total_frames;
      for (int i = 0; i < total_frames; i++)
      {
          //if (i % total_frames == 0)
          //{
          //    cap_.set(CV_CAP_PROP_POS_FRAMES, 0);
          //}
          cap_ >> tmpMat;
          imgCache.push_back(tmpMat.clone());
      }
  }
  vector<vector<struct detect_result>> objects;
  double warm_up_time = 0;
  double detect_time = 0;
  // Initialize the network.
  if (use_fp16) {
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
    detector.Detect(objects, iter, visualize);
    detect_timer.Stop();
    detect_time = detect_timer.MilliSeconds();
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
    detector.Detect(objects, iter, visualize);
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

