#include "caffe/imagenet_mean.hpp"

//switch RGB to BGR indexing (for Caffe convention)
inline int get_BGR(int channel_RGB) { assert( channel_RGB < 3 ); return 2 - channel_RGB; }


// get avg value for imagenet pixels on particular channel
// these are defined in caffe/include/caffe/imagenet_mean.hpp
inline float get_mean_RGB(int channel_RGB){
  if(channel_RGB==0) { return float(IMAGENET_MEAN_R); }
  if(channel_RGB==1) { return float(IMAGENET_MEAN_G); }
  if(channel_RGB==2) { return float(IMAGENET_MEAN_B); }
  else { assert(0); }
  return 0;
}

template< typename T >
inline std::string str(T const & i)	// convert T i to string
{
  std::stringstream s;
  s << i;
  return s.str();
}
