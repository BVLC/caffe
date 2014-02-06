#include "caffe/imagenet_mean.hpp"
#include "boost/shared_ptr.hpp"
#include "caffe/caffe.hpp"
#include <vector>
using namespace std;
using boost::shared_ptr;
using namespace caffe;

//switch RGB to BGR indexing (for Caffe convention)
inline int get_BGR(int channel_RGB) { assert( channel_RGB < 3 ); return 2 - channel_RGB; }

class DenseNet_Params{
    public:
    //TODO: create a constructor or '.setDefaults()' that defines default params

    int interval; // # scales per octave (default = 10)
    int imwidth;
    int imheight;
    int padx;
    int pady;
    int sbin; //convnet_subsampling_ratio. = 16 for alexnet conv5 features.

    //doesn't include 'scales', because 'scales' may be a different data type for matlab vs python

};

typedef vector< float > vect_float;
typedef shared_ptr< vect_float > p_vect_float;
typedef vector< p_vect_float > vect_p_vect_float;

static void raw_do_forward( shared_ptr<Net<float> > net_, vect_p_vect_float const & bottom ) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(bottom.size(), input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    assert( bottom[i]->size() == uint32_t(input_blobs[i]->count()) );
    const float* const data_ptr = &bottom[i]->front();
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  net_->ForwardPrefilled();
}

template< typename T > inline std::string str(T const & i) { std::stringstream s; s << i; return s.str(); } // convert T i to string
