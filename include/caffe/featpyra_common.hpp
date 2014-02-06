#include "caffe/imagenet_mean.hpp"

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

template< typename T > inline std::string str(T const & i) { std::stringstream s; s << i; return s.str(); } // convert T i to string
