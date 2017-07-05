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

#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
using namespace caffe;  // NOLINT(build/namespaces)

#define Dtype half

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file);

  void Detect(cv::Mat& img);

 private:
  void WrapInputLayer(std::vector<Dtype *> &input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<Dtype *> input_channels);

  shared_ptr<Net<Dtype> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
    int count = 0;
#ifndef CPU_ONLY
    count = Caffe::EnumerateDevices(true);
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
}


Detector::Detector(const string& model_file,
                   const string& weights_file) {
  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
#ifndef CPU_ONLY
    std::cout << "Use GPU with device ID " << gpus[0] << std::endl;
    //Caffe::SetDevices(gpus);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpus[0]);
#endif  // !CPU_ONLY
  } else {
    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);
  }

  /* Load the network. */
  net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Detector::Detect(cv::Mat& img) {
  int w = img.cols;
  int h = img.rows;
  Blob<Dtype>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<Dtype *> input_channels;
  WrapInputLayer(input_channels);

  Preprocess(img, input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<Dtype>* result_blob = net_->output_blobs()[0];
  const Dtype* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  for (int k = 0; k < num_det * 7; k+=7) {
	int imgid = (int)result[k+0];
	int classid = (int)result[k+1];
	float confidence = result[k+2];
    int left = (int)((result[k+3]-result[k+5]/2.0) * w);
    int right = (int)((result[k+3]+result[k+5]/2.0) * w);
    int top = (int)((result[k+4]-result[k+6]/2.0) * h);
    int bot = (int)((result[k+4]+result[k+6]/2.0) * h);
	cv::rectangle(img,cvPoint(left,top),cvPoint(right,bot),cv::Scalar(255, 242, 35));
	std::stringstream ss;
	ss << classid << "/" << confidence;
	std::string  text = ss.str();
	cv::putText(img, text, cvPoint(left,top+20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<Dtype *> &input_channels) {
  Blob<Dtype>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  Dtype* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    input_channels.push_back(input_data);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<Dtype *> input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::divide(sample_float, 255.0, sample_normalized);

  for( int_tp i = 0; i < input_geometry_.height; i++) {
    for( int_tp j = 0; j < input_geometry_.width; j++) {
      int pos = i * input_geometry_.width + j;
      if (num_channels_ == 3) {
        cv::Vec3f pixel = sample_normalized.at<cv::Vec3f>(i, j);
        input_channels[0][pos] = pixel.val[0];
        input_channels[1][pos] = pixel.val[1];
        input_channels[2][pos] = pixel.val[2];
      } else {
        cv::Scalar pixel = sample_normalized.at<float>(i, j);
        input_channels[0][pos] = pixel.val[0];
      }
    }
  }
}

int main(int argc, char** argv) {

  if (argc < 3) {
    return 1;
  }
  std::streambuf* buf = std::cout.rdbuf();
  std::ostream out(buf);
  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& filename = argv[3];
  const string& filenameout = argv[4];

  // Initialize the network.
  Detector detector(model_file, weights_file);

  cv::Mat img = cv::imread(filename, -1);
  Timer detect_timer;
  detect_timer.Start();
  double timeUsed;
  detector.Detect(img);
  detect_timer.Stop();
  timeUsed = detect_timer.MilliSeconds();
  out << "lym time=" << timeUsed/1000 <<"ms\n";

  cv::imwrite(filenameout, img);

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV and half floating point support."
             << "compile with USE_OPENCV and USE_ISAAC.";
}
#endif  // USE_OPENCV
