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

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Timer;

#define Dtype float

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
  cv::Size input_newwh_;  
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

  Timer detect_timer;
  detect_timer.Start();
  double timeUsed;

  net_->Forward();
  
  detect_timer.Stop();
  timeUsed = detect_timer.MilliSeconds();
  std::cout << "forward time=" << timeUsed <<"ms\n";

  /* Copy the output layer to a std::vector */
  Blob<Dtype>* result_blob = net_->output_blobs()[0];
  const Dtype* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  for (int k = 0; k < num_det*7; k+=7) {
	int imgid = (int)result[k+0];
	int classid = (int)result[k+1];
	float confidence = result[k+2];
	int left=0,right,top=0,bot;
	if(input_newwh_.width==0){
	    left = (int)((result[k+3]-result[k+5]/2.0) * w);
	    right = (int)((result[k+3]+result[k+5]/2.0) * w);
	    top = (int)((result[k+4]-result[k+6]/2.0) * h);
	    bot = (int)((result[k+4]+result[k+6]/2.0) * h);
	}
	else{
        left =  w*(result[k+3] - (input_geometry_.width - input_newwh_.width)/2./input_geometry_.width)*input_geometry_.width / input_newwh_.width; 
        top =  h*(result[k+4] - (input_geometry_.height - input_newwh_.height)/2./input_geometry_.height)*input_geometry_.height / input_newwh_.height; 
        float boxw = result[k+5]*w*input_geometry_.width/input_newwh_.width;
        float boxh= result[k+6]*h*input_geometry_.height/input_newwh_.height; 
        left-=(int)(boxw/2);
        top-=(int)(boxh/2);
        right = (int)(left+boxw);
        bot=(int)(top+boxh);
	}
    if (left < 0)
        left = 0;
    if (right > w-1)
        right = w-1;
    if (top < 0)
        top = 0;
    if (bot > h-1)
        bot = h-1;
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
    cv::cvtColor(img, sample, cv::COLOR_BGRA2RGB);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
  else
    sample = img;

  cv::Mat sample_resized;
  int dx=0,dy=0;
  input_newwh_ = input_geometry_;
  if (sample.size() != input_geometry_){
    int netw = input_geometry_.width;
	int neth = input_geometry_.height;
	int width = sample.cols;
	int height = sample.rows;
	if(width!=height){ //if img is not square, must fill the img at center
	    if ((netw*1.0/width) < (neth*1.0/height)){
	        input_newwh_.width= netw;
	        input_newwh_.height = (height * netw)/width;
	    }
	    else{
	        input_newwh_.height = neth;
	        input_newwh_.width = (width * neth)/height;
	    }
	    dx=(netw-input_newwh_.width)/2;
	    dy=(neth-input_newwh_.height)/2;  
		cv::resize(sample, sample_resized, input_newwh_);
	}
	else
    	cv::resize(sample, sample_resized, input_geometry_);
  }
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::divide(sample_float, 255.0, sample_normalized);
  if(dx!=0 || dy!=0) {
    for(int i=0;i< num_channels_;i++) {
      for(int pos = 0; pos < input_geometry_.width*input_geometry_.height; ++pos) {
        input_channels[i][pos] = 0.5;
      }
    }
  }

  for( int i = 0; i < input_newwh_.height; i++) {
    for( int j = 0; j < input_newwh_.width; j++) {
      int pos = (i+dy) * input_geometry_.width + j+dx;
      if (num_channels_ == 3) {
        cv::Vec3f pixel = sample_normalized.at<cv::Vec3f>(i, j);
        input_channels[0][pos] = pixel.val[2];
        input_channels[1][pos] = pixel.val[1];
        input_channels[2][pos] = pixel.val[0];  //RGB2BGR
      } else {
        cv::Scalar pixel = sample_normalized.at<float>(i, j);
        input_channels[0][pos] = pixel.val[0];
      }
    }
  }
  if(dx==0 && dy==0)
  	input_newwh_.width=0; //clear the flag
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
  out << "the first detect time=" << timeUsed <<"ms\n";

  img = cv::imread(filename, -1);
  detect_timer.Start();
  detector.Detect(img);
  detect_timer.Stop();
  timeUsed = detect_timer.MilliSeconds();
  out << "the second detect time=" << timeUsed <<"ms\n";
  
  
  cv::imwrite(filenameout, img);

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
