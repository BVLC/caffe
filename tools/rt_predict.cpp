#include <cuda_runtime.h>
 
#include <cstring>
#include <cstdlib>
#include <vector>
 
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/algorithm/string.hpp>
 
using namespace caffe;
using namespace std;
using namespace cv;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");

DEFINE_int32(classnum, -1,
    "The class of label");

DEFINE_string(image, "",
    "test image path");

// Parse GPU ids or use all available devices
void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}


bool check_args() {
    if ( FLAGS_model.size()==0 || FLAGS_classnum == -1 || FLAGS_weights.size() == 0)
        return false;
    return true;
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: rt_predict <command> <args>\n\n"
      "commands:\n"
      "  model           model path\n"
      "  weight          trained weights\n"
      "  device_query    show GPU diagnostic information\n"
      "  image           test image\n");
  // Run tool or show usage.
 
    caffe::GlobalInit(&argc, &argv);

    if (!check_args()) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/rt-predict");
        cout<<FLAGS_image<<" "<<FLAGS_model<<endl;
        exit(1);
    }
    device_query(); 
    //get the net
    Net<float> caffe_test_net(FLAGS_model,caffe::TEST);
    //get trained net
    caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);
    cv::Mat image = imread(FLAGS_image,CV_LOAD_IMAGE_COLOR);
    std::vector<cv::Mat> dv;
    dv.push_back(image);
    std::vector<int> dvl;
    dvl.push_back(0);
    // Run ForwardPrefilled 
    float loss = 0.0;
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            caffe_test_net.layers()[0])->AddMatVector(dv,dvl);
    const vector<Blob<float>*>& result =  caffe_test_net.Forward(&loss);
     
    // Now result will contain the argmax results.
    const float* argmaxs = result[2]->cpu_data();
    cout<<"result 0 len : "<<result[0]->num()<<" "<<result[0]->count()<<" "<<result.size()<< endl;
    for (int i = 0; i < 8; ++i) {
        LOG(INFO) << " Image: "<< i << " class:" << argmaxs[i];
    }
  
    return 0;
}
