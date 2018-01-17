#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

const int BLANK_LABEL = 10;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             int blank_label);

  std::vector<int> Classify(const cv::Mat& img);

 private:
  std::vector<int> Predict(const cv::Mat& img);
  void GetLabelseqs(const std::vector<int>& label_seq_with_blank,
                    std::vector<int>& label_seq);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  int blank_label_;
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       int blank_label) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  blank_label_ = blank_label;
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Classifier::GetLabelseqs(const std::vector<int>& label_seq_with_blank,
                              std::vector<int>& label_seq) {
  label_seq.clear();
  int prev = blank_label_;
  int length = label_seq_with_blank.size();
  for(int i = 0; i < length; ++i) {
    int cur = label_seq_with_blank[i];
    if(cur != prev && cur != blank_label_) {
      label_seq.push_back(cur);
    }
    prev = cur;
  }
}
/* Return the top N predictions. */
std::vector<int> Classifier::Classify(const cv::Mat& img) {
  std::vector<int> pred_label_seq_with_blank = Predict(img);
  std::vector<int> pred_label_seq;
  GetLabelseqs(pred_label_seq_with_blank, pred_label_seq);
  return pred_label_seq;
}

std::vector<int> Classifier::Predict(const cv::Mat& img) {
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
  Blob<float>* output_layer = net_->output_blobs()[0];

  const int time_step = input_geometry_.width;
  const int alphabet_size = output_layer->shape(2);

  std::vector<int> pred_label_seq_with_blank(time_step);
  const float* pred_data = output_layer->cpu_data();

  for(int t = 0; t < time_step; ++t) {
    pred_label_seq_with_blank[t] = std::max_element(pred_data, pred_data + alphabet_size) - pred_data;
    pred_data += alphabet_size;
  }

  return pred_label_seq_with_blank;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void Classifier::Preprocess(const cv::Mat& img,
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

  sample_float /= 255.0;

  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img.jpg/.png(or other OpenCV supported format)" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];

  Classifier classifier(model_file, trained_file, BLANK_LABEL);

  string file = argv[3];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<int> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  std::cout << "digits: ";
  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << predictions[i];
  }
  std::cout << std::endl;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
