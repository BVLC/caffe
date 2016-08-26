#include <cuda_runtime.h>
 
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
#include <string>
#include <iostream>
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
#include <omp.h>
 
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

void crop_image_memory(vector<vector<cv::Mat> > & crop_set,std::string image_path) {
    cv::Mat image = imread(image_path,CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        LOG(ERROR)<< " imread "<<image_path<<" error";
    }
    // now it's hardcode for jpeg encode
    int img_y = image.size().height;
    int img_x = image.size().width;
    crop_set.resize(100);

    #pragma omp parallel for
    for (size_t i= 1; i<= 100 ; ++i ) {
        vector<int> param(2);
        param[0] = CV_IMWRITE_JPEG_QUALITY;
        param[1] = i;
        vector<uchar> buff;
        vector<cv::Mat> crop_jpeg_set;
        imencode(".jpg",image,buff,param);
        cv::Mat image_jpeg = imdecode(Mat(buff),CV_LOAD_IMAGE_COLOR);
        cv::Rect roi;
        roi.width = FLAGS_width;
        roi.height = FLAGS_height;

        for (int x = 0 ; x <= img_x-FLAGS_height ; x+= FLAGS_height) {
            for (int y = 0; y <= img_y -FLAGS_width ; y+=FLAGS_width) {
                roi.x = x;
                roi.y = y;
                crop_jpeg_set.push_back(image_jpeg(roi));
            }
        }
        crop_set[i-1] = crop_jpeg_set;
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

std::vector<float> crop_image_predict_prob(Net<float>& caffe_test_net,
                        const std::vector<cv::Mat>& tmpSlice,
                        int bsize)
{
        float loss = 0.0;
        std::vector<float> predictResult(FLAGS_classnum,0.0);
        std::vector<int> dvl(tmpSlice.size(),0);
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                caffe_test_net.layers()[0])->set_batch_size(bsize);
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                caffe_test_net.layers()[0])->AddMatVector(tmpSlice,dvl);

        const vector<Blob<float>*>& result =  caffe_test_net.Forward(&loss);
        const float* argmaxs = result[2]->cpu_data();
        for(size_t j = 0; j<bsize ; ++j) {
            const float* p_argmaxs = argmaxs + j*FLAGS_classnum;
            for(size_t p = 0 ; p<FLAGS_classnum ; ++p) {
                predictResult[p]+= *(p_argmaxs+p) ;
            }
        }
        return predictResult;
}

void image_tsigma_distribution(const vector<float>& y)
{
    size_t y_len = y.size();
    float mu = std::accumulate(y.begin(),y.end(),0.0)/y_len;
    vector<float> y_squre_error(y_len);
    for (size_t i = 0; i<y_len; ++i) {
        y_squre_error[i] = (pow(y[i]-mu,2.0));
    }
    float max_jnd_value = *std::max_element(y.begin(),y.end());
    float min_jnd_value = *std::min_element(y.begin(),y.end());

    float sigma = sqrt(std::accumulate(y_squre_error.begin(),
                                       y_squre_error.end(), 0.0)/y_len)/(max_jnd_value - min_jnd_value);
    float slice = (max_jnd_value - min_jnd_value) / (FLAGS_classnum - 1);
    vector<int> jnd_predict(FLAGS_classnum);
    jnd_predict[0] = 100;
    for (size_t i = 1; i<FLAGS_classnum ; ++i) {
        //float tsigma = mu - sigma;
        float tsigma = min_jnd_value + i*slice;
        size_t min_index = -1;
        float  min_dist = 0;
        for (size_t j = 0 ; j<y_len; ++j) {
            float dist = fabs(y[j] - tsigma);
            if( min_index == -1 ) {
                min_dist = dist;
                min_index = j;
            }
            else if( min_dist > dist ) {
                min_index = j;
                min_dist = dist;
            }
            jnd_predict[i] = min_index+1;
        }
    }
    LOG(INFO)<<"mu = "<<mu<<", sigma = "<<sigma;
    for (size_t i = 0; i<FLAGS_classnum; ++i) {
        LOG(INFO)<<"JND="<<i<<",predicted value : "<<jnd_predict[i];
    }

//    return tsigma;

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
    if(FLAGS_gpu.size()){
        caffe::Caffe::set_mode(Caffe::GPU);
        device_query(); 
    } else {
        caffe::Caffe::set_mode(Caffe::CPU);
    }
    int max_threads_num = omp_get_max_threads();
    vector<Net<float>* > caffe_test_net_set(max_threads_num);

    Net<float> caffe_test_net(FLAGS_model,caffe::TEST);
    caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);

    vector<vector<Mat> > crop_set; 
    crop_image_memory(crop_set,FLAGS_image);
    const float labelOrder[] = {0,1,2,3,4,5,6,7};
    int bsize = FLAGS_batchsize;
    vector<float> jnd_jpeg(100);

    //#pragma omp parallel for
    for(size_t factor = 0 ; factor < crop_set.size(); factor++ ) {
        if(!(factor%10)) {
            LOG(INFO)<<"Now "<<factor<<"/100";
        }
        vector<Mat> dv = crop_set[factor];
        float* predictResult = new float[FLAGS_classnum];
        std::fill_n(predictResult,FLAGS_classnum,0.0);
        size_t roiSize = dv.size();
        int opMax = roiSize/bsize;
        
        for (size_t i = 0 ; i<opMax; i++ ) {
            //LOG(INFO)<<" bsize : "<<i<<"/"<<opMax;
            vector<Mat> tmpSlice(dv.begin()+(i)*bsize,dv.begin()+(i+1)*bsize);
            vector<float> tmpProb = 
                crop_image_predict_prob(
                        caffe_test_net,
                        tmpSlice,
                        bsize);
            for(size_t p = 0; p< FLAGS_classnum ; ++p) {
                predictResult[p]+=tmpProb[p];
            }
        }
        for(size_t i=0; i<FLAGS_classnum ; ++i) {
            predictResult[i]/=roiSize;
            //LOG(INFO)<<"i="<<i<<" prob="<<predictResult[i];
        }
        jnd_jpeg[factor] = caffe::caffe_cpu_dot(FLAGS_classnum,labelOrder,predictResult);
        //LOG(INFO)<<"factor "<<factor<<",predict : "<< jnd_jpeg[factor];
        delete[] predictResult;
    }
    for(size_t i = 0; i<crop_set.size() ;++i) {
        LOG(INFO)<<"JPEG QoF="<<i+1<<",predict="<<jnd_jpeg[i];
    }
    image_tsigma_distribution(jnd_jpeg);

    return 0;
}
