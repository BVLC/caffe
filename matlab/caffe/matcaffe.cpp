// Copyright Ross Girshick and Yangqing Jia 2013
//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include "mex.h"
#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;

// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet
{
  // The pointer to the internal caffe::Net instance
	shared_ptr<Net<float> > net_;

  CaffeNet() {}
  
  void init(string param_file, string pretrained_param_file) {
    net_.reset(new Net<float>(param_file));
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  virtual ~CaffeNet() {}

  /*
  inline void check_array_against_blob(
      PyArrayObject* arr, Blob<float>* blob) {
    CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
    CHECK_EQ(PyArray_NDIM(arr), 4);
    CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
    npy_intp* dims = PyArray_DIMS(arr);
    CHECK_EQ(dims[0], blob->num());
    CHECK_EQ(dims[1], blob->channels());
    CHECK_EQ(dims[2], blob->height());
    CHECK_EQ(dims[3], blob->width());
  }
  */

  // Data needs to be [images, channels, height, width] where width is the fastest dimension
  // 
  // In matlab, reading an image gives [height, width, channels] where height is the fastest dimension
  //  - want to have the order as [width, height, channels, images]
  //    (channels in BGR order)
  //  - 
  //
  // The matlab model is: 
  //   - bottom is a cell array of 4D tensors in the correct format
  //   - top is allocated in here as a cell array of outputs
  //
  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and c contiguous.
  //
  //
  mxArray* Forward(const mxArray* const bottom) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]), 
        input_blobs.size());
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      const mxArray* const elem = mxGetCell(bottom, i);
      const float* const data_ptr = 
          reinterpret_cast<const float* const>(mxGetPr(elem));
      //check_array_against_blob(arr, input_blobs[i]);
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
      //check_array_against_blob(arr, output_blobs[i]);
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

};

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
  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id); 
}

static CaffeNet net;

static void net_init(MEX_ARGS) {
  net.init("/home/rbg/working/caffe/examples/imagenet_deploy.prototxt", 
           "/home/rbg/working/caffe/examples/alexnet_train_iter_470000");
}

static void net_forward(MEX_ARGS) {
  plhs[0] = net.Forward(prhs[0]);
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
  { "forward",            net_forward     },
  { "init",               net_init        },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  // TODO: check args
  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    //bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        //dispatched = true;
        break;
      }
    }
    mxFree(cmd);
    //checkM(dispatched, "Command not found!");
  }
}
