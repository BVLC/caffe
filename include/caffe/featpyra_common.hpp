#include "caffe/imagenet_mean.hpp"
#include "boost/shared_ptr.hpp"
#include "caffe/caffe.hpp"
#include <vector>

namespace caffe {

  using namespace std;
  using boost::shared_ptr;

//switch RGB to BGR indexing (for Caffe convention)
  inline int get_BGR(int channel_RGB) { assert( channel_RGB < 3 ); return 2 - channel_RGB; }

  struct densenet_params_t {
    uint32_t interval; // # scales per octave
    uint32_t img_padding; // in image pixels (at scale=1.0). could later allow per-dim img_padx/img_pady as alternative.
    densenet_params_t( void ) { // default values
      interval = 10;
      img_padding = 16;
    }
  };

  typedef vector< float > vect_float;
  typedef shared_ptr< vect_float > p_vect_float;
  typedef vector< p_vect_float > vect_p_vect_float;

  typedef vector< uint32_t > vect_uint32_t;
  typedef shared_ptr< vect_uint32_t > p_vect_uint32_t;

  // a sketch of a possible shared output type for python/matlab interfaces. missing dims for feats.
  struct densenet_output_t {
    p_vect_float imwidth; // not including padding
    p_vect_float imheight;
    p_vect_uint32_t feat_padx;
    p_vect_uint32_t feat_pady;
    p_vect_float scales;
    vect_p_vect_float feats;
    uint32_t nb_planes;
  };

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
}
