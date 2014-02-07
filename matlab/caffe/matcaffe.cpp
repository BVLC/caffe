// Copyright Ross Girshick and Yangqing Jia 2013
//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <string>
#include <vector>

#include "mex.h"
#include "caffe/caffe.hpp"
#include "stitch_pyramid/PyramidStitcher.h" //also includes JPEGImage, Patchwork, etc
#include "boost/shared_ptr.hpp"
#include "caffe/featpyra_common.hpp"
#include<stdexcept>

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [batch_images, channels, height, width]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.
static mxArray* do_forward(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
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
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    mxArray* mx_blob = mxCreateNumericMatrix(output_blobs[i]->count(),
        1, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, output_blobs[i]->cpu_data(),
          sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
          sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

template< typename T >
T sz_from_dims( uint32_t const num_dims, T const * const dims ) {
  T ret = 1;
  for( uint32_t dim = 0; dim < num_dims; ++dim ) { ret *= dims[dim]; }
  return ret;
}

typedef vector< mxArray * > vect_rp_mxArray;
//typedef vector< float > vect_float; // now defined in featpyra_common.hpp
//typedef shared_ptr< vect_float > p_vect_float;
//typedef vector< p_vect_float > vect_p_vect_float;

p_vect_float make_p_vect_float( size_t const num ) {
  p_vect_float ret( new vect_float );
  ret->resize( num, 0.0f );
  return ret;
};

/*
static void raw_do_forward( vect_p_vect_float const & bottom ) {
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
*/

static p_vect_float copy_output_blob_data( uint32_t const output_blob_ix )
{

  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  if( ! (output_blob_ix < output_blobs.size() ) ) {
    LOG(FATAL) << "!(output_blobs_ix < output_blobs.size())";
  }  
  Blob<float> * const output_blob = output_blobs[output_blob_ix];

  int batchsize = net_->output_blobs()[0]->num();
  int depth = net_->output_blobs()[0]->channels();
  int width = net_->output_blobs()[0]->width();
  int height = net_->output_blobs()[0]->height();
  mwSize dims[4] = {batchsize, width, height, depth};

  if( sz_from_dims( 4, dims ) != (uint32_t)output_blob->count() ) {
    LOG(FATAL) << "sz_from_dims( 4, dims ) != output_blob->count()";
  }
  p_vect_float ret = make_p_vect_float( (uint32_t)output_blob->count() );
  float * const dest = &ret->front();
  switch (Caffe::mode()) {
  case Caffe::CPU: memcpy(dest, output_blob->cpu_data(), sizeof(float) * ret->size() ); break;
  case Caffe::GPU: cudaMemcpy(dest, output_blob->gpu_data(), sizeof(float) * ret->size(), cudaMemcpyDeviceToHost); break;
  default: LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())
  return ret;
}


// The caffe::Caffe utility functions.
static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_forward(prhs[0]);
}


void check_dims_equal( uint32_t const num_dims, uint32_t const * const dims_a, uint32_t const * const dims_b ) {
  bool dims_eq = 1;
  for( uint32_t dim = 0; dim < num_dims; ++dim ) { if( dims_a[dim] != dims_b[dim] ) { dims_eq = 0; } }
  if( !dims_eq ) { throw( std::runtime_error( "dims unequal" ) ); }
}

void check_input_blobs_dims( uint32_t const num_dims, uint32_t const * const dims_b )
{
  if( num_dims != 4 ) { throw( std::runtime_error( "wrong # dims" ) ); }
  int batchsize = net_->input_blobs()[0]->num();
  int depth = net_->input_blobs()[0]->channels();
  int width = net_->input_blobs()[0]->width();
  int height = net_->input_blobs()[0]->height();
  uint32_t dims[4] = {batchsize, depth, height, width};
  check_dims_equal( 4, dims, dims_b );
}

p_vect_float p_vect_float_from_output_blob_0( void ) {
  if( net_->num_outputs() != 1 ) { 
    LOG(FATAL) << "expecting 1 output blob, but got " << net_->num_outputs();
  }
  int batchsize = net_->output_blobs()[0]->num();
  if( batchsize != 1 ) {
    LOG(FATAL) << "expecting batchsize=1, but got batchsize=" << batchsize;
  }
  return copy_output_blob_data( 0 );
}

p_vect_float JPEGImage_to_p_float( JPEGImage &jpeg ){
  int depth = jpeg.depth();
  int height = jpeg.height();
  int width = jpeg.width();
  int batchsize = 1;
  uint32_t dims[4] = {batchsize, depth, height, width};
  check_input_blobs_dims( 4, dims );

  uint32_t const ret_sz = sz_from_dims( 4U, dims );
  p_vect_float ret = make_p_vect_float( ret_sz );
  
  //copy jpeg into jpeg_float_npy
  for(int ch_src=0; ch_src<depth; ch_src++){ //ch_src is in RGB
    int ch_dst = get_BGR(ch_src); //for Caffe BGR convention
    //float const ch_mean = get_mean_RGB(ch_src); //mean of all imagenet pixels of this channel
    float const ch_mean = IMAGENET_MEAN_RGB[ch_src];
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
	//jpeg:           row-major, packed RGB, RGB, ...          uint8_t.
	//rp_float: row-major, unpacked BBBB..,GGGG..,RRRR.. float.
	uint32_t const rix = ch_dst*height*width+y*width+x;
	ret->at(rix) = jpeg.bits()[y*width*depth + x*depth + ch_src] - ch_mean;
      }
    }
  }
  return ret;
}

// @param out output a list of mxArray *'s, one list element per scale, each holding a 3D numeric array of the features
// @param scaleLocs = location of each scale on planes (see unstitch_pyramid_locations in PyramidStitcher.cpp)
// @param descriptor_planes -- each element of the list is a plane of Caffe descriptors
//          typically, descriptor_planes = blobs_top.
// @param depth = #channels (typically 256 for conv5)
static mxArray * unstitch_planes( vector<ScaleLocation> const & scaleLocs, vect_p_vect_float const & descriptor_planes, int depth) {
  uint32_t const nbScales = scaleLocs.size();
  mxArray * const ret = mxCreateCellMatrix( nbScales, 1 );
  assert( ret );

  for(uint32_t i=0; i<nbScales; i++) //go from largest to smallest scale
  { 
    uint32_t depth = net_->output_blobs()[0]->channels();
    uint32_t width = net_->output_blobs()[0]->width();
    uint32_t height = net_->output_blobs()[0]->height();

    int planeID = scaleLocs[i].planeID;
    assert( uint32_t(planeID) < descriptor_planes.size() );
    p_vect_float dp = descriptor_planes[planeID];

    uint32_t const dp_dims[3] = {width,height,depth};

    // row-major / C / numpy / caffe dims (note: the matlab dims of descriptor_planes are (correctly) the reverse of this)
    // dims[3] = {depth, height, width}; 

    mwSize ret_dims[3] = {depth, scaleLocs[i].height, scaleLocs[i].width }; // desired column-major / F / matlab dims
    mxArray * const ret_scale = mxCreateNumericArray( 3, ret_dims, mxSINGLE_CLASS, mxREAL );
    float * const ret_scale_data = (float * )mxGetData( ret_scale );
    mwSize ret_sz = sz_from_dims( 3, ret_dims );
    for( uint32_t x = 0; x < uint32_t(ret_dims[2]); ++x ) {
      for( uint32_t y = 0; y < uint32_t(ret_dims[1]); ++y ) {
	for( uint32_t d = 0; d < uint32_t(ret_dims[0]); ++d ) {
	  uint32_t const rix = d + y*ret_dims[0] + x*ret_dims[0]*ret_dims[1];
	  assert( rix < uint32_t(ret_sz) );
	  uint32_t const dp_x = x + scaleLocs[i].xMin;
	  uint32_t const dp_y = y + scaleLocs[i].yMin;
	  uint32_t const dp_ix = dp_x + dp_y*dp_dims[0] + d*dp_dims[0]*dp_dims[1];
	  //ret_scale_data[rix] = float(d) + 1000.0*y + 1000000.0*x;
	  ret_scale_data[rix] = dp->at(dp_ix);
	}
      }
    }
    mxSetCell( ret, i, ret_scale );
  }
  return ret;
}


static uint32_t mx_to_u32( string const & err_str, mxArray const * const mxa ) {
  if( !mxIsNumeric( mxa ) ) { mexErrMsgTxt( (err_str + ": expected a numeric value, got something else. ").c_str()); }
  uint32_t const sz = mxGetNumberOfElements( mxa );
  if( sz != 1 ) { mexErrMsgTxt( (err_str+": expected a single element, but got " + str(sz) + ".").c_str()); }
  return uint32_t(mxGetScalar(mxa));
}

static mxArray * u32_to_mx( uint32_t const val ) { // returned as float (i.e. scalar single)
  mxArray * const mxa = mxCreateNumericMatrix( 1, 1, mxSINGLE_CLASS, mxREAL );
  *(float*)(mxGetData(mxa)) = (float)val;
  return mxa;
}

char const * fnames[] = { "scales", "feat", "imwidth", "imheight", "feat_padx", "feat_pady" };

static void convnet_featpyramid(MEX_ARGS) {
  if ( (nrhs < 1) || (nrhs > 2) ) {
    LOG(ERROR) << "Given " << nrhs << " arguments, expected 1 or 2.";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (nlhs != 1) {
    LOG(ERROR) << "Caller wanted " << nlhs << " outputs, but this function always produces 1.";
    mexErrMsgTxt("Wrong number of outputs");
  }
  char *fn_cs = mxArrayToString(prhs[0]);
  if( !fn_cs ) { mexErrMsgTxt("Could not convert first argument to a string."); }
  string const file( fn_cs );
  mxFree(fn_cs);
  
  densenet_params_t params; // ctor sets params to defaults

  if( nrhs > 1 ) { // (try to) parse second arg as params
    mxArray const * const mx_params = prhs[1];
    if( !mxIsStruct( mx_params ) ) { 
      mexErrMsgTxt("Expected second argument to be a struct, but it was not."); }
    uint32_t const npes = mxGetNumberOfElements( mx_params );
    if( npes != 1 ) {
      mexErrMsgTxt(("Expected second argument to be a struct with one element, but it had " + str(npes) + ".").c_str()); }
    uint32_t const nf = mxGetNumberOfFields( mx_params );
    for( uint32_t fi = 0; fi != nf; ++fi ) {
      char const * const fn = mxGetFieldNameByNumber( mx_params, fi );
      assert( fn );
      mxArray const * const mx_f = mxGetFieldByNumber( mx_params, 0, fi );
      assert( mx_f );
      if( 0 ) { }
      else if( !strcmp( fn, "interval" ) ) { params.interval = mx_to_u32( "for param interval", mx_f ); }
      else if( !strcmp( fn, "img_padding" ) ) { params.img_padding = mx_to_u32( "for param img_padding", mx_f ); }
      else { mexErrMsgTxt( ("unknown parameter " + string(fn) ).c_str() ); }
    }
  }

  int convnet_subsampling_ratio = 16; //for conv5 layer features
  int planeDim = net_->input_blobs()[0]->width(); //assume that all preallocated blobs are same size
  int resultDepth = net_->output_blobs()[0]->channels();

  assert(net_->input_blobs()[0]->width() == net_->input_blobs()[0]->height()); //assume square planes in Caffe. (can relax this if necessary)
  assert(net_->input_blobs()[0]->num() == 1); //for now, one plane at a time.)
  //TODO: verify/assert that top-upsampled version of input img fits within planeDim

  Patchwork patchwork = stitch_pyramid(file, params.img_padding, params.interval, planeDim); 
  int nbPlanes = patchwork.planes_.size();

  vect_p_vect_float blobs_top;
  //prep input data for Caffe feature extraction    
  for(int planeID=0; planeID<nbPlanes; planeID++){
    vect_p_vect_float blobs_bottom; //input buffer(s) for Caffe::Forward 
    blobs_bottom.push_back( JPEGImage_to_p_float(patchwork.planes_.at(planeID)) ); 
    //raw_do_forward( blobs_bottom ); //lists of blobs... bottom[0]=curr input planes, top_tmp[0]=curr output descriptors
    raw_do_forward( net_, blobs_bottom ); 
    blobs_top.push_back( p_vect_float_from_output_blob_0() );
  }

  vector<ScaleLocation> scaleLocations = unstitch_pyramid_locations(patchwork, convnet_subsampling_ratio);
  uint32_t const ret_rows = patchwork.scales_.size();
  assert( scaleLocations.size() == ret_rows );

  mxArray * const feats = unstitch_planes( scaleLocations, blobs_top, resultDepth );
  mxArray * const scale = mxCreateNumericMatrix( ret_rows, 1, mxSINGLE_CLASS, mxREAL );
  float * const scale_ptr = (float*)(mxGetData(scale));
  for( uint32_t r = 0; r < ret_rows; ++r ) { scale_ptr[r] = patchwork.scales_[r]; }

  mxArray * ret = mxCreateStructMatrix( 1, 1, sizeof(fnames)/sizeof(char*), fnames ); // see fnames for field names
  mxSetFieldByNumber( ret, 0, 0, scale );
  mxSetFieldByNumber( ret, 0, 1, feats );
  mxSetFieldByNumber( ret, 0, 2, u32_to_mx( patchwork.imwidth_ ) );
  mxSetFieldByNumber( ret, 0, 3, u32_to_mx( patchwork.imheight_ ) );
  mxSetFieldByNumber( ret, 0, 4, u32_to_mx( 1 ) );
  mxSetFieldByNumber( ret, 0, 5, u32_to_mx( 1 ) );

  plhs[0] = ret;
}


/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "init",               init            },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  // featpyramid functions 
  { "convnet_featpyramid",convnet_featpyramid },

  // The end.
  { "END",                NULL            },
};

// building with mkoctfile
// cd ~/git_work/caffe ;  CXXFLAGS="-Wall -fpic -O2" mkoctfile --mex matlab/caffe/matcaffe.cpp libcaffe.a -pthread -I/usr/local/include -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -Wall -L/usr/lib -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/opt/intel/mkl/lib -L/opt/intel/mkl/lib/intel64 -lcudart -lcublas -lcurand -lprotobuf -lopencv_core -lopencv_highgui -lglog -lmkl_rt -lmkl_intel_thread -lleveldb -lsnappy -lpthread -lboost_system -lopencv_imgproc -L/home/moskewcz/git_work/caffe/python/caffe/stitch_pyramid -lPyramidStitcher -I./python/caffe -o matlab/caffe/caffe

/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
