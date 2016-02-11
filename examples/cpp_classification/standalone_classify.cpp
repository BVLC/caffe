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
#include <future>
#include <thread>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <boost/range.hpp>
#include <boost/circular_buffer.hpp>
#include <math.h>       /* fabs */

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

enum TrashClass {
  COMPOST = 0,
  RECYCLE = 1,
  LANDFILL = 2,
  ELECTRONICS = 3
};

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 4);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

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

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::cout << "Before loading the labels" << std::endl;
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  std::cout << "After loading model" << std::endl;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
  std::cout << "Finished loading lable file" << std::endl;

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
  std::cout << "Done initializing" << std::endl;
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
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
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
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

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

std::pair<float, float> mean_and_variance(const boost::circular_buffer<float>& xs) {
    // begin(xs) will point to the first element in xs
    // end(xs) will point to the last element in xs
    // the distance between them gives the number of elements
    size_t N = end(xs) - begin(xs);
    // first pass through all data (hidden in accumulate):
    float m = std::accumulate(begin(xs), end(xs), 0.0) / N;
    float s2 = 0;
    // second pass through all data:
    for(auto x : xs) {
        s2 += (x - m) * (x - m);
    }
    return std::pair<float, float>(m, s2 / (N-1));
}

std::string toPrettyText(TrashClass trashClass) {
  switch (trashClass) {
    case RECYCLE:
      return "It should go to recycle";
    case COMPOST:
      return "It should go to compost";
    case LANDFILL:
      return "It should go to landfill";
    case ELECTRONICS:
      return "Electronics bin (building 17)";
    default:
      return "No object found";
  }
}

void print_prediction(
    const std::string& prediction,
    TrashClass trashClass,
    cv::Mat& image) {
  cv::Scalar color;
  switch(trashClass) {
    case RECYCLE: 
      // blue
      color = cv::Scalar(255, 0, 0, 150);
      break;
    case COMPOST: 
      // green
      color = cv::Scalar(0, 255, 0, 150);
      break;
    case LANDFILL: 
      // green
      color = cv::Scalar(30, 30, 30, 150);
      break;
    case ELECTRONICS: 
      // red
      color = cv::Scalar(0, 0, 255, 150);
      break;
  }

  cv::rectangle(
    image,
    cv::Point2f(0, 0),
    cv::Point2f(600, 220),
    color,
    -1
  );

  cv::putText(
      image,
      prediction,
      cv::Point2f(50, 100),
      cv::FONT_HERSHEY_SIMPLEX,
      2,
      cv::Scalar(255, 255, 255, 255),
      4);
  cv::putText(
      image,
      toPrettyText(trashClass), 
      cv::Point2f(50, 160),
      cv::FONT_HERSHEY_SIMPLEX,
      1,
      cv::Scalar(255, 255, 255, 255),
      2);
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  std::unordered_map<std::string, TrashClass> labelToTrashClass {
    {"banana", COMPOST},
    {"bottle", RECYCLE},
    {"beer bottle", RECYCLE},
    {"burrito", COMPOST},
    {"carton", RECYCLE},
    {"cellular telephone, cellular phone, cellphone, cell, mobile phone", ELECTRONICS},
    {"coffee cup", RECYCLE},
    {"cup", RECYCLE},
    {"custard apple", COMPOST},
    {"pot, flowerpot", LANDFILL},
    {"Granny Smith", COMPOST},
    {"hair spray", LANDFILL},
    {"hand-held computer, hand-held microcomputer", ELECTRONICS},
    {"iPod", ELECTRONICS},
    {"computer keyboard, keypad", ELECTRONICS},
    {"mouse, computer mouse", ELECTRONICS},
    {"packet, packaging", LANDFILL},
    {"paper towel", RECYCLE},
    {"pizza, pizza pie", COMPOST},
    {"plastic bag", LANDFILL},
    {"pomegranate", COMPOST},
    {"pop bottle, soda bottle", RECYCLE},
    {"soccer ball", LANDFILL},
    {"spaghetti squash", COMPOST},
    {"water bottle", RECYCLE},
    {"wine bottle", RECYCLE},
    {"vase", LANDFILL},
  };

  ::google::InitGoogleLogging(argv[0]);
  const int RLV_SIZE = 5;
  int frame_skipped = 0;

  // mutex for launching tasks
  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  std::cout << "Trying to open the camera" << std::endl;
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    LOG(FATAL) << "Cannot open camera!";   
  }

  std::cout << "Camera opened, trying to open a new window" << std::endl;
  cv::namedWindow("Display", cv::WINDOW_NORMAL);
  cv::setWindowProperty("Display", cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

  std::mutex g_image_lock;
  cv::Mat global_image; 

  std::mutex g_prediction_lock;
  std::vector<Prediction> prediction_results;
  std::cout << "Trying to start async task" << std::endl;

  // Running Laplatian variance of buffer size RLV_SIZE
  boost::circular_buffer<float> rlv(RLV_SIZE);

  std::thread t([&classifier, &g_image_lock, &global_image, &g_prediction_lock, &prediction_results, &rlv, &frame_skipped] () {
    while(1) {
    std::cout << "starting async task" << std::endl;
      if (global_image.empty()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      g_image_lock.lock();
      cv::Mat image_local = global_image;
      g_image_lock.unlock();

      cv::Mat lap;
      cv::Laplacian(image_local, lap, CV_64F);
      cv::Scalar mu, sigma;
      cv::meanStdDev(lap, mu, sigma);

      float var = sigma.val[0] * sigma.val[0];

      // If running buffer is too small, accummulate some more
      if (rlv.size() < RLV_SIZE) {
        rlv.push_back(var);
        continue;
      }
      
      std::pair<float, float> meanAndVar = mean_and_variance(rlv);
      std::cout << "Show mean " << meanAndVar.first << " and variance " << meanAndVar.second 
        << " and fabs " << fabs(var - meanAndVar.first)
        << " and current variance " << var
        << std::endl;
      rlv.push_back(var);
      
      // If not much has changed in the history, just keep skipping
      if (meanAndVar.second < (meanAndVar.first / 5.0)) {
        continue;
      }
      if (fabs(var - meanAndVar.first) > meanAndVar.second) {
        continue;
      }

      // something significant and we need to scan, so make sure we skip a few
      frame_skipped++;
      if (frame_skipped < 2) {
        continue;
      }
      frame_skipped = 0;

      std::vector<Prediction> predictions = classifier.Classify(image_local);

      // Print the top N predictions.
      std::cout << "==================" << std::endl;
      for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
      }
      g_prediction_lock.lock();
      prediction_results = predictions;
      g_prediction_lock.unlock();
    }
  });

  std::cout << "Detaching " << std::endl;
  t.detach();
  std::cout << "Detached " << std::endl;
  std::pair<std::string, TrashClass> previousResult;

  while(1) {
    cv::Mat image;
    cap >> image;

    if (image.empty()) {
      std::cout << "Unable to decode image ";
      sleep(1);
      continue;
    }

    bool hasLock = g_image_lock.try_lock();
    if (hasLock) {
      global_image = image.clone();
      g_image_lock.unlock();
    }

    g_prediction_lock.lock();
    std::vector<Prediction> local_prediction = prediction_results;
    g_prediction_lock.unlock();
    for (size_t i = 0; i < local_prediction.size(); i++) {
      if (local_prediction[i].second < 0.1) {
        continue;
      }
      auto it = labelToTrashClass.find(local_prediction[i].first);
      if (it != labelToTrashClass.end()) {
        print_prediction(local_prediction[i].first, it->second, image);
        previousResult = std::make_pair(local_prediction[i].first, it->second);
        break;
      } 

      // At this point, which means it has not been broken out of the loop
      // show cached result once in case it is a blurred frame
      if (!previousResult.first.empty()) {
        print_prediction(previousResult.first, previousResult.second, image);
        previousResult.first.clear();
      }
    }
    cv::imshow("Display", image);
    char key = cv::waitKey(1);
    if (key == 'q') {
      exit(0);
    }
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
