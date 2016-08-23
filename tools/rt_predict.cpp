#include <cuda_runtime.h>
 
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
 
#include <string>
#include <iostream>
#include <stdio.h>
//#include <future>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
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
DEFINE_int32(height,64,
    "crop image height");
DEFINE_int32(width,64,
    "crop image width");
DEFINE_int32(batchsize,1,
    "batch size in test phase");

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

void crop_image_memory(std::vector<cv::Mat>& crop_set,std::string image_path) {
    cv::Mat image = imread(image_path,CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        LOG(ERROR)<< " imread "<<image_path<<" error";
    }
    // now it's hardcode for jpeg encode
    vector<int> param(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 51;
    vector<uchar> buff;
    imencode(".jpg",image,buff,param);
    image = imdecode(Mat(buff),CV_LOAD_IMAGE_COLOR);

    cv::Rect roi;
    roi.width = FLAGS_width;
    roi.height = FLAGS_height;
    int img_y = image.size().height;
    int img_x = image.size().width;
    for (int x = 0 ; x <= img_x-FLAGS_height ; x+= FLAGS_height) {
        for (int y = 0; y <= img_y -FLAGS_width ; y+=FLAGS_width) {
            roi.x = x;
            roi.y = y;
            crop_set.push_back(image(roi));
        }
    }
}
std::vector<int> crop_image_predict(Net<float>& caffe_test_net,
                        const std::vector<cv::Mat>& tmpSlice,
                        int bsize)
{
        float loss = 0.0;
        std::vector<int> predictResult;
        std::vector<int> dvl(tmpSlice.size(),0);
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                caffe_test_net.layers()[0])->AddMatVector(tmpSlice,dvl);
        const vector<Blob<float>*>& result =  caffe_test_net.Forward(&loss);
        const float* argmaxs = result[2]->cpu_data();
        for(size_t j = 0; j<bsize ; ++j) {
            const float* p_argmaxs = argmaxs + j*FLAGS_classnum;
            int argMax = std::distance(p_argmaxs, std::max_element(p_argmaxs,p_argmaxs+FLAGS_classnum));
            predictResult.push_back(argMax);
        }
        return predictResult;
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
    if(FLAGS_gpu.size())
        device_query(); 
    else 
        caffe::Caffe::set_mode(Caffe::CPU);

    //get the net
    Net<float> caffe_test_net(FLAGS_model,caffe::TEST);
    //get trained net
    caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);
    std::vector<cv::Mat> dv ;
    crop_image_memory(dv,FLAGS_image);
    std::vector<int> predictResult;
    int bsize = FLAGS_batchsize;
    int opMax = dv.size()/bsize;
    for (size_t i = 0 ; i<opMax; i+=3 ) {
        LOG(INFO)<<" bsize : "<<i<<"/"<<opMax;
        vector<vector<int> > tmpResult(3);
        for (size_t j = 0; j<3 ; j++){
            std::vector<cv::Mat> tmpSlice(dv.begin()+(i+j)*bsize,dv.begin()+(i+1+j)*bsize);
            //auto handle = std::async(crop_image_predict,caffe_test_net,tmpSlice,bsize);
            tmpResult[j] = crop_image_predict(caffe_test_net,tmpSlice,bsize);
            predictResult.insert(predictResult.end(),tmpResult[j].begin(),tmpResult[j].end());
        }

    }
    vector<int> countRank(FLAGS_classnum,0);

    for (size_t i = 0 ; i < predictResult.size() ; ++i) {
        countRank[predictResult[i]]++;
    }

    for(size_t i = 0 ; i< FLAGS_classnum ; ++i) {
        LOG(INFO) << " i : " << i<<" "<<countRank[i];
    }

    int maxRank = std::distance(countRank.begin(),
                                std::max_element(countRank.begin(),countRank.end()));
    LOG(INFO) << "predict JND : " << maxRank;

    return 0;
}
