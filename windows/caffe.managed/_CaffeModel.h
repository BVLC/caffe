// Due to a bug caused by C++/CLI and boost (used indirectly via caffe headers, not this one), 
// we have to seperate code related to boost from CLI compiling environment.
// This wrapper class serves for this purpose.
// See: http://stackoverflow.com/questions/8144630/mixed-mode-c-cli-dll-throws-exception-on-exit
//	and http://article.gmane.org/gmane.comp.lib.boost.user/44515/match=string+binding+invalid+mixed

#pragma once

#include <string>
#include <vector>

//Declare an abstract Net class instead of including caffe headers, which include boost headers.
//The definition of Net is defined in cpp code, which does include caffe header files.
namespace caffe
{
	template <class DType>
	class Net;
}

struct FloatArray
{
  const float* Data;
  int Size;
  FloatArray(const float* data, int size);
};

typedef std::vector<float> FloatVec;

class _CaffeModel
{
	caffe::Net<float>* _net;

public:
  static void SetDevice(int device_id); //Use a negative number for CPU only

  _CaffeModel(const std::string &netFile, const std::string &modelFile);
	~_CaffeModel();

  //REVIEW ktran: these APIs only make sense for images
	FloatArray ExtractOutputs(const std::string &imageFile, int interpolation, const std::string &layerName);
  std::vector<FloatArray> ExtractOutputs(const std::string &imageFile, int interpolation, const std::vector<std::string> &layerNames);
};
