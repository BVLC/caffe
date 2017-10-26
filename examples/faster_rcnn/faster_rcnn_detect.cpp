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
#include <caffe/caffe.hpp>

#if defined(USE_OPENCV)
using namespace caffe;  // NOLINT(build/namespaces)

struct detect_result {
  int imgid;
  int classid;
  float confidence;
  float left;
  float right;
  float top;
  float bottom;
};

template <typename Dtype>
class Detector {
public:
  Detector(const string& model_file,
           const string& weights_file,
           int gpu);
  void Preprocess(const string& img_file);

  void Detect(int iter_count = 1,
              bool visualize = false);

  void ShowResult(const cv::Mat &imgs,
                  const vector<detect_result>& objects);
  ~Detector() {
    for (int i = 0; i < input_blobs_.size(); i++)
      delete input_blobs_[i];
  }

private:
  shared_ptr<Net<Dtype> > net_;
  cv::Size input_blob_size_;
  vector<Blob<Dtype>*> input_blobs_;
  cv::Mat ori_img;  //original image without resize
};

template <typename Dtype>
Detector<Dtype>::Detector(const string& model_file,
                   const string& weights_file,
                   int gpu) {
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

  /* Load the network. */
  net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<Dtype>* input_data = net_->input_blobs()[0];
  CHECK(input_data->channels() == 3);
}

static void
FixupChannels(cv::Mat& img) {
  if (img.channels() != 3) {
    cv::Mat sample;
    if (img.channels() == 4)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else {
      // Should not enter here, just in case.
      sample = img;
    }
    img = sample;
  }
}

template <typename Dtype>
void Detector<Dtype>::Preprocess(const string& img_file) {

  ori_img = cv::imread(img_file, -1);
  FixupChannels(ori_img);

  //ref to https://github.com/rbgirshick/caffe-fast-rcnn
  cv::Mat img_float;
  ori_img.convertTo(img_float, CV_32FC3);

  img_float = img_float - cv::Scalar(102.9801, 115.9465, 122.7717);

  static int MAX_SIZE = 1000;
  static int SCALE_SIZE = 600;
  int img_size_min = std::min(ori_img.rows, ori_img.cols);
  int img_size_max = std::max(ori_img.rows, ori_img.cols);
  float scale = SCALE_SIZE * 1.0 / img_size_min;
  if (scale * img_size_max > MAX_SIZE)
    scale = MAX_SIZE * 1.0 / img_size_max;

  cv::Size image_scaled_size;
  image_scaled_size.width = (int)(scale * ori_img.cols);
  image_scaled_size.height = (int)(scale * ori_img.rows);

  cv::Mat img_resized;
  if (image_scaled_size.width != ori_img.cols)
    cv::resize(img_float, img_resized, image_scaled_size);
  else
    img_resized = img_float;

  Blob<Dtype> * blob = new Blob<Dtype>;
  blob->Reshape(1,
                3,
                img_resized.rows,
                img_resized.cols);
  Dtype* data = blob->mutable_cpu_data();
  Dtype* firstC = data;
  Dtype* secondC = data + blob->offset(0, 1, 0, 0);
  Dtype* thirdC = data + blob->offset(0, 2, 0, 0);
  for( int i = 0; i < blob->height(); i++) {
    for( int j = 0; j < blob->width(); j++) {
      int pos = i * blob->width() + j;
      cv::Vec3f pixel = img_resized.at<cv::Vec3f>(i, j);
      firstC[pos] = pixel.val[0];
      secondC[pos] = pixel.val[1];
      thirdC[pos] = pixel.val[2];
    }
  }
  input_blobs_.push_back(blob);

  blob = new Blob<Dtype>;
  blob->Reshape(std::vector<int>{1, 3});
  data = blob->mutable_cpu_data();
  data[0] = img_resized.rows;
  data[1] = img_resized.cols;
  data[2] = scale;
  input_blobs_.push_back(blob);
}

const char* classNames[] = {"__background__",
                            "aeroplane",
                            "bicycle",
                            "bird",
                            "boat",
                            "bottle",
                            "bus",
                            "car",
                            "cat",
                            "chair",
                            "cow",
                            "diningtable",
                            "dog",
                            "horse",
                            "motorbike",
                            "person",
                            "pottedplant",
                            "sheep",
                            "sofa",
                            "train",
                            "tvmonitor"};

template <typename Dtype>
void Detector<Dtype>::ShowResult(const cv::Mat &img,
                                 const vector<detect_result>& objects)
{
  if (objects.size() == 0)
    return;

  cv::Mat show_img = img.clone();

  for (int i = 0; i < objects.size(); i++) {
    detect_result obj = objects[i];
    cv::rectangle(show_img,
                    cvPoint(obj.left, obj.top),
                    cvPoint(obj.right, obj.bottom),
                    cv::Scalar(255, 242, 35));
    std::stringstream ss;
    ss << classNames[obj.classid] << "/" << obj.confidence;
    std::string  text = ss.str();
    cv::putText(show_img,
                     text,
                     cvPoint(obj.left, obj.top + 20),
                     cv::FONT_HERSHEY_PLAIN,
                     1.0f,
                     cv::Scalar(0, 255, 255));
  }

  cv::imshow("detections", show_img);
  int key = cv::waitKey(static_cast<char>(0));
  if (key == 'q')
    exit(1);
}

template <typename Dtype>
void Detector<Dtype>::Detect(int iter_count,
                             bool visualize) {
  for (int i = 0; i < iter_count; ++i) {
    Blob<Dtype>* imdata = net_->input_blobs()[0];
    imdata->ReshapeLike(*input_blobs_[0]);
    imdata->ShareData(*input_blobs_[0]);

    Blob<Dtype>* iminfo = net_->input_blobs()[1];
    iminfo->ReshapeLike(*input_blobs_[1]);
    iminfo->ShareData(*input_blobs_[1]);

    net_->Reshape();
    net_->Forward();
  }

  Blob<Dtype>* result_blob = net_->output_blobs()[0];
  std::cout << result_blob->shape_string() << std::endl;
  const Dtype* result = result_blob->cpu_data();
  const int num_det = result_blob->height();

  vector<detect_result> objects;
  for (int k = 0; k < num_det * 7; k += 7) {
    detect_result object;
    object.imgid = (int)result[k + 0];
    object.classid = (int)result[k + 1];
    object.confidence = result[k + 2];
    object.left = (int)(result[k + 3]);
    object.top = (int)(result[k + 4]);
    object.right = (int)(result[k + 5]);
    object.bottom = (int)(result[k + 6]);
    objects.push_back(object);
  }
  if (visualize)
    ShowResult(ori_img, objects);
}

int main(int argc, char** argv) {

#if CV_MAJOR_VERSION >= 3
  const char* keys =
    "{ model model_file      | <none> | model file }"
    "{ weights weights_file  | <none> | weights file }"
    "{ img image             | <none> | path to image }"
    "{ v visualize           | false  | visualize output }"
    "{ i iter                | 1      | iterations to be run }"
    "{ g gpu                 | 0      | gpu device }"
    "{ c cpu                 | false  | use cpu device }"
    "{ fp16 use_fp16         | false  | use fp16 forward engine. }"
    "{ help                  | false  | display this help and exit  }"
    ;
#else
  const char* keys =
    "{ model   | model_file      | <none> | model file }"
    "{ weights | weights_file    | <none> | weights file }"
    "{ img     | image           | <none> | path to image }"
    "{ v       | visualize       | false  | visualize output }"
    "{ i       | iter            | 1      | iterations to be run }"
    "{ g       | gpu             | 0      | gpu device }"
    "{ c       | cpu             | false  | use cpu device }"
    "{ fp16    | use_fp16        | false  | use fp16 forward engine. }"
    "{ h       | help            | false  | display this help and exit  }"
    ;
#endif

  cv::CommandLineParser parser(argc, argv, keys);
  string model_file = parser.get<std::string>("model_file");
  string weights_file = parser.get<std::string>("weights_file");
  string img_file = parser.get<std::string>("image");
  bool visualize = parser.get<bool>("visualize");
  int iter = parser.get<int>("iter");
  int gpu = parser.get<int>("gpu");
  bool cpu = parser.get<bool>("cpu");
  bool use_fp16 = parser.get<bool>("use_fp16");

  if (model_file == "" ||
      weights_file == "" ||
      img_file == "") {
#if CV_MAJOR_VERSION >= 3
    parser.printMessage();
#else
    parser.printParams();
#endif
    exit(-1);
  }

  if (cpu)
    gpu = -1;

  double warm_up_time = 0;
  double detect_time = 0;
  // Initialize the network.
  if (use_fp16) {
#ifdef HAS_HALF_SUPPORT
    Detector<half> detector(model_file, weights_file, gpu);
    Timer warm_up_timer, detect_timer;
    detector.Preprocess(img_file);
    // warm up.
    warm_up_timer.Start();
    //detector.Detect();
    //detector.Detect();
    warm_up_timer.Stop();
    warm_up_time = warm_up_timer.MilliSeconds();

    detect_timer.Start();
    detector.Detect(iter, visualize);
    detect_timer.Stop();
    detect_time = detect_timer.MilliSeconds();
#else
    std::cout << "fp16 is not supported." << std::endl;
#endif
  } else {
    Detector<float> detector(model_file, weights_file, gpu);
    Timer warm_up_timer, detect_timer;
    detector.Preprocess(img_file);
    // warm up.
    warm_up_timer.Start();
    //detector.Detect();
    //detector.Detect();
    warm_up_timer.Stop();
    warm_up_time = warm_up_timer.MilliSeconds();

    detect_timer.Start();
    detector.Detect(iter, visualize);
    detect_timer.Stop();
    detect_time = detect_timer.MilliSeconds();
  }

  std::cout << "Warm up time is "
            << warm_up_time
            << "ms." << std::endl;

  std::cout << "Detect time is "
            << detect_time
            << "ms." << std::endl;

  double fps = iter / (detect_time / 1000.);
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
