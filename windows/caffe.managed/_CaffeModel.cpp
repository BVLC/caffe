#include "_CaffeModel.h"

#pragma warning(push, 0) // disable warnings from the following external headers
#include <vector>
#include <string>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#pragma warning(push, 0) 

using namespace boost;
using namespace caffe;

const cv::Scalar_<float> BlackPixel(0, 0, 0);

FloatArray::FloatArray(const float* data, int size) : Data(data), Size(size) {}

_CaffeModel::_CaffeModel(const string &netFile, const string &modelFile)
{
  _net = new Net<float>(netFile, Phase::TEST);
  _net->CopyTrainedLayersFrom(modelFile);
}

_CaffeModel::~_CaffeModel()
{
	if (_net)
	{
		delete _net;
		_net = nullptr;
	}
}

void _CaffeModel::SetDevice(int deviceId)
{
  // Set GPU
  if (deviceId >= 0)
  {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(deviceId);
  }
  else
    Caffe::set_mode(Caffe::CPU);
}

int _CaffeModel::GetInputImageWidth()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->width();
}

int _CaffeModel::GetInputImageHeight()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->height();
}

int _CaffeModel::GetInputImageChannels()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->channels();
}

cv::Mat CVReadImage(const string &imageFile, int height, int width, int interpolation)
{
  float means[3] = { 103.939, 116.779, 123.68 }; //REVIEW ktran: why hardcoded and why is it useful?

	cv::Mat cv_img_origin = cv::imread(imageFile);
  if (!cv_img_origin.data)
    throw std::runtime_error("Could not open or find file");

	cv::Mat cv_img_float(cv_img_origin.rows, cv_img_origin.cols, CV_32FC3);

	float *dst = &cv_img_float.at<cv::Vec3f>(0, 0)[0];
	unsigned char *src = &cv_img_origin.at<cv::Vec3b>(0, 0)[0];
	int imgSize = cv_img_origin.rows * cv_img_origin.cols;
	for (int x = 0; x < imgSize; ++x)
		for (int c = 0; c < 3; ++c)
			*dst++ = static_cast<float>(*src++) - means[c];

	cv::Mat resizedImg;
	double maxSize = std::max(cv_img_origin.rows, cv_img_origin.cols);
  resize(cv_img_float, resizedImg, cv::Size(), double(width) / maxSize, double(height) / maxSize, interpolation);
  
  if (resizedImg.cols < width)
  {
    cv::Mat blankImg(height, width - resizedImg.cols, CV_32FC3, BlackPixel);
    hconcat(resizedImg, blankImg, resizedImg);
  }
  else
    resizedImg.resize(height, BlackPixel);
    
  assert(resizedImg.rows == height && resizedImg.cols == width);

  return resizedImg;
}

void Evaluate(caffe::Net<float>* net, const string &imageFile, int interpolation)
{
  //Step 1: read image and prefill the input blob
  Blob<float>* input_blob = net->input_blobs()[0];
  int height = input_blob->height();
  int width = input_blob->width();
  cv::Mat img = CVReadImage(imageFile, height, width, interpolation);
  
  float* src_data = &img.at<cv::Vec3f>(0, 0)[0];	// h*w*c
  float* input_data = input_blob->mutable_cpu_data();
  for (int c = 0; c < 3; ++c)
  {
    for (int h = 0; h < height; ++h)
    {
      float *src = src_data + h * width * 3 + c;
      for (int w = 0; w < width; ++w, ++input_data)
        *input_data = src[w * 3];
    }
  }

  //Step 3: run a forward pass
  float loss = 0.0;
  net->ForwardPrefilled(&loss);
}

FloatArray _CaffeModel::ExtractOutputs(const string &imageFile, int interpolation, const string &blobName)
{
  Evaluate(_net, imageFile, interpolation);
  auto blob = _net->blob_by_name(blobName);
  return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractOutputs(const string &imageFile, int interpolation, const vector<string> &layerNames)
{
  Evaluate(_net, imageFile, interpolation);
  vector<FloatArray> results;
  for (auto& name : layerNames)
  {
    auto blob = _net->blob_by_name(name);
    results.push_back(FloatArray(blob->cpu_data(), blob->count()));
  }
  return results;
}

void EvaluateBitmap(caffe::Net<float>* net, const string &imageData, int interpolation)
{
    // Net initialization
    float loss = 0.0;
    shared_ptr<MemoryDataLayer<float> > memory_data_layer;
    memory_data_layer = static_pointer_cast<MemoryDataLayer<float>>(net->layer_by_name("data"));

    Datum datum;
    datum.set_channels(3);
    datum.set_height(memory_data_layer->height());
    datum.set_width(memory_data_layer->width());
    datum.set_label(0);
    datum.clear_data();
    datum.clear_float_data();
    datum.set_data(imageData);

    std::vector<Datum> datums;
    for (int i = 0; i < 1; i++)
        datums.push_back(datum);

    memory_data_layer->AddDatumVector(datums);
    const std::vector<Blob<float>*>& results = net->ForwardPrefilled(&loss);

}

FloatArray _CaffeModel::ExtractBitmapOutputs(const std::string &imageData, int interpolation, const string &blobName)
{
    EvaluateBitmap(_net, imageData, interpolation);
    auto blob = _net->blob_by_name(blobName);
    return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractBitmapOutputs(const std::string &imageData, int interpolation, const vector<string> &layerNames)
{
    EvaluateBitmap(_net, imageData, interpolation);
    vector<FloatArray> results;
    for (auto& name : layerNames)
    {
        auto blob = _net->blob_by_name(name);
        results.push_back(FloatArray(blob->cpu_data(), blob->count()));
    }
    return results;
}