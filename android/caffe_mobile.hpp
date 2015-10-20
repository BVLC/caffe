#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
<<<<<<< HEAD
#include <vector>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>

using std::string;
using std::vector;

namespace caffe {

class CaffeMobile {
public:
  ~CaffeMobile();

  static CaffeMobile *Get();
  static CaffeMobile *Get(const string &model_path, const string &weights_path);

  void SetMean(const string &mean_file);

  void SetMean(const vector<float> &mean_values);

  void SetScale(const float scale);

  vector<int> PredictTopK(const string &img_path, int k);

  vector<vector<float>> ExtractFeatures(const string &img_path,
                                        const string &str_blob_names);

private:
  static CaffeMobile *caffe_mobile_;
  static string model_path_;
  static string weights_path_;

  CaffeMobile(const string &model_path, const string &weights_path);

  void Preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels);

  void WrapInputLayer(std::vector<cv::Mat> *input_channels);

  vector<float> Forward(const string &filename);

  shared_ptr<Net<float>> net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  float scale_;
=======
#include "caffe/caffe.hpp"

using std::string;

namespace caffe {

class CaffeMobile
{
public:
	CaffeMobile(string model_path, string weights_path);
	~CaffeMobile();

	int test(string img_path);

	vector<int> predict_top_k(string img_path, int k=3);

private:
	Net<float> *caffe_net;
>>>>>>> 25d8ecc... Added jni lib for android
};

} // namespace caffe

#endif
