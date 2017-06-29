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
#include <sys/time.h>  

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file);

  void Detect(cv::Mat& img);

 private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
 
  shared_ptr<Net<float> > net_;
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
    Caffe::SetDevices(gpus);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpus[0]);
#endif  // !CPU_ONLY
  } else {
    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);
  }

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST, Caffe::GetDefaultDevice()));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Detector::Detect(cv::Mat& img) {
  int w = img.cols;
  int h = img.rows;
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  for (int k = 0; k < num_det; k+=7) {
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
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
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

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
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
      struct timeval tstart,tend;  
	double timeUsed;  
	gettimeofday(&tstart, NULL);  	  
    detector.Detect(img);
	gettimeofday(&tend, NULL);  
    timeUsed=1000000*(tend.tv_sec-tstart.tv_sec)+tend.tv_usec-tstart.tv_usec;  
	out << "lym time=" << timeUsed/1000 <<"ms\n";
  
  cv::imwrite(filenameout, img);

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
