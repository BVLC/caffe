//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
// used by default unless a net reference is passed in
static shared_ptr<Net<float> > net_;
static int init_key = -2;

// Storage for nets by reference
std::map<uint16_t, shared_ptr<Net<float> > > nets_;
pthread_mutex_t mutex_add_net_;

/** -----------------------------------------------------------------
 ** net handle management
 **/

static uint16_t add_net(shared_ptr<Net<float> >* net, uint16_t handle = 0) {
    static uint16_t handle_ctr = 0;

    if (handle < 1) {
        pthread_mutex_lock(&mutex_add_net_);
        uint16_t next_handle = ++handle_ctr;
        pthread_mutex_unlock(&mutex_add_net_);

        nets_[next_handle] = (*net);

        CHECK_GE(next_handle, 1);
        return next_handle;
    } else {
        nets_[handle] = (*net);
        return handle;
    }
}

static shared_ptr<Net<float> > get_net(const uint16_t handle = 0) {
    if (handle == 0) {
        return net_;
    } else {
        std::map<uint16_t, shared_ptr<Net<float> > >::iterator it;
        it = nets_.find(handle);

        if (it == nets_.end()) {
            mexErrMsgTxt("Invalid net handle");
        }

        shared_ptr<Net<float> > net = it->second;
        CHECK(net) << "Returned null pointer for lookup: " << handle;

        return net;
    }
}

static void remove_net(const uint16_t handle) {
    CHECK_GE(handle, 1);

    std::map<uint16_t, shared_ptr<Net<float> > >::iterator it;
    it = nets_.find(handle);

    if (it == nets_.end()) {
        mexErrMsgTxt("Invalid net handle");
    }

    nets_.erase(it);
}

static uint16_t get_handle(const mxArray* array) {
    if (!mxIsUint16(array)) {
        mexErrMsgTxt("Handle should be of type uint16_t");
    }
    uint16_t* handle_arr = static_cast<uint16_t*>(mxGetData(array));
    return handle_arr[0];
}

/** -----------------------------------------------------------------
 ** main module functions
 **/

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
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

static mxArray* do_forward(const mxArray* const bottom,
                           const std::vector<std::string>& blob_names =
                             std::vector<std::string>(),
                           shared_ptr<Net<float> > net = net_) {
  if (!net) {
    mex_error("Calling 'forward' on an uninitialized net - call 'init' first");
  }

  const vector<Blob<float>*>& input_blobs = net->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      std::string error_msg;
      error_msg += "MatCaffe input size does not match the input size ";
      error_msg += "of the network";
      mex_error(error_msg);
    }

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  // do forward
  const vector<Blob<float>*>& output_blobs = net->ForwardPrefilled();

  // pointer to store blobs for return
  const vector<Blob<float>*>* result_blobs = 0;

  const bool retrieve_custom_blobs = (!blob_names.empty());
  vector<Blob<float>*> extract_blobs(blob_names.size());

  if (!retrieve_custom_blobs) {
    // by default, only output blobs will be returned
    result_blobs = &output_blobs;
  } else {
    for (size_t i = 0; i < blob_names.size(); ++i) {
      const shared_ptr<Blob<float> > custom_output_blob =
        net->blob_by_name(blob_names[i]);
      extract_blobs[i] = const_cast<Blob<float>*>(custom_output_blob.get());
    }
    // but get pointers to requested custom output blobs if specified
    result_blobs = &extract_blobs;
  }

  mxArray* mx_out = mxCreateCellMatrix(result_blobs->size(), 1);

  for (unsigned int i = 0; i < result_blobs->size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    const vector<Blob<float>*>& result_blobs_ref = *result_blobs;

    mwSize dims[4] = {result_blobs_ref[i]->width(),
                      result_blobs_ref[i]->height(),
                      result_blobs_ref[i]->channels(),
                      result_blobs_ref[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(result_blobs_ref[i]->count(), result_blobs_ref[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(result_blobs_ref[i]->count(), result_blobs_ref[i]->gpu_data(),
          data_ptr);
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())

    mxSetCell(mx_out, i, mx_blob);
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff,
                            shared_ptr<Net<float> > net = net_) {
  const vector<Blob<float>*>& output_blobs = net->output_blobs();
  const vector<Blob<float>*>& input_blobs = net->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) !=
      output_blobs.size()) {
    mex_error("Invalid input size");
  }
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights(shared_ptr<Net<float> > net = net_) {
  const vector<shared_ptr<Layer<float> > >& layers = net->layers();
  const vector<string>& layer_names = net->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {
  uint16_t handle = 0;
  if ((nrhs >= 1) && (!mxIsEmpty(prhs[0]))) {
    handle = get_handle(prhs[0]);
  }
  shared_ptr<Net<float> > net = get_net(handle);

  plhs[0] = do_get_weights(net);
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if ((nrhs < 3) || (nrhs > 5)) {
    ostringstream error_msg;
    error_msg << "Wrong number of arguments. Usage: caffe('init', "
              << "model_def_file, model_file, phase_name,"
              << "[, input_count=<0=from model>, net_handle=<0>])"
              << "Where net_handle=N and "
              << "N=0 (use singleton global net instance), "
              << "N=-1 (create new net instance and return handle), "
              << "N>0 (reinitialize net instance with handle N)";
    mex_error(error_msg.str());
  }

  const int32_t GLOBAL_NET = 0;
  const int32_t CREATE_NET = -1;

  // Handle input params
  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);
  char* phase_name = mxArrayToString(prhs[2]);

  size_t input_count = 0;
  int32_t net_handle = 0;
  if (nrhs >= 4) {
    input_count = mxGetScalar(prhs[3]);
  }
  if (nrhs >= 5) {
    net_handle = static_cast<int32_t>(mxGetScalar(prhs[4]));
  }

  // Get phase
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mex_error("Unknown phase.");
  }

  // Handle output params

  if (((net_handle == GLOBAL_NET) || (net_handle > 0)) && (nlhs != 0))
    mexErrMsgTxt("0 output arugments if net_handle>=0"
                 "(use global net/existing handle)");
  if (net_handle == CREATE_NET) {
    if (nlhs != 1) {
      LOG(ERROR) << "Must provide output arugment if using net_handle<0"
                 << " to store the created net pointer";
      mexErrMsgTxt("Wrong number of output arugments");
    }
  }

  LOG(INFO) << "Starting instantiation...";
  // Instantiate a new net
  if (net_handle == GLOBAL_NET) {
    LOG(INFO) << "Instantiating global net...";
    net_.reset(new Net<float>(string(param_file), phase, input_count));
    net_->CopyTrainedLayersFrom(string(model_file));

    init_key = random();  // NOLINT(caffe/random_fn)

    if (nlhs == 1) {
      plhs[0] = mxCreateDoubleScalar(init_key);
    }
  } else if (net_handle == CREATE_NET) {
    shared_ptr<Net<float > >
      net(new Net<float>(string(param_file),
                         phase, input_count));
    net->CopyTrainedLayersFrom(string(model_file));

    uint16_t* created_handle;
    mwSize dims[3] = {1, 1};
    plhs[0] = mxCreateNumericArray(2, dims, mxUINT16_CLASS, mxREAL);
    created_handle = static_cast<uint16_t*>(mxGetData(plhs[0]));

    (*created_handle) = add_net(&net);
  } else if (net_handle > 0) {
    shared_ptr<Net<float > >
      net(new Net<float>(string(param_file),
                         phase, input_count));
    net->CopyTrainedLayersFrom(string(model_file));

    add_net(&net, net_handle);

    uint16_t* return_handle;
    mwSize dims[3] = {1, 1};
    if (nlhs == 1) {
      plhs[0] = mxCreateNumericArray(2, dims, mxUINT16_CLASS, mxREAL);
      return_handle = static_cast<uint16_t*>(mxGetData(plhs[0]));
      (*return_handle) = net_handle;
    }
  } else {
    mex_error("Unrecognized value for net_handle");
  }

  mxFree(param_file);
  mxFree(model_file);
  mxFree(phase_name);
}

static void reset(MEX_ARGS) {
  if ((nrhs != 0) || (nrhs != 1)) {
    mex_error("Wrong number of arguments. "
              "Usage: caffe('reset'[, net_handle=<common>]");
  }

  uint16_t handle = 0;
  if ((nrhs >= 1) && (!mxIsEmpty(prhs[0]))) {
    handle = get_handle(prhs[0]);
  }
  shared_ptr<Net<float> > net = get_net(handle);

  if (net) {
    net.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if ((nrhs < 1) || (nrhs > 3)) {
    ostringstream error_msg;
    error_msg << "Wrong number of arguments. Usage: caffe('forward', images"
              << "[, blob_names=<[]=final>, net_handle=<common>])";
    mex_error(error_msg.str());
  }

  // Handle params
  std::vector<std::string> blob_names;
  uint16_t handle = 0;

  if ((nrhs >= 2) && (!mxIsEmpty(prhs[1]))) {
    if (!mxIsCell(prhs[1])) {
      mex_error("blob_names should be a cell array of strings");
    }
    size_t blob_count = mxGetNumberOfElements(prhs[1]);
    for (size_t i = 0; i < blob_count; ++i) {
      blob_names.push_back(std::string(mxArrayToString(mxGetCell(prhs[1], i))));
    }
  }

  if ((nrhs >= 3) && (!mxIsEmpty(prhs[2]))) {
    handle = get_handle(prhs[2]);
  }

  shared_ptr<Net<float> > net = get_net(handle);
  if (!net) {
    mex_error("Net was uninitialized in call to 'forward'");
  }

  // Do forward
  plhs[0] = do_forward(prhs[0], blob_names, net);
}

static void backward(MEX_ARGS) {
  if ((nrhs < 1) && (nrhs > 2)) {
    mex_error("Wrong number of arguments. "
              "Usage: caffe('backward', top_diff"
              "[, net_handle=<common>])");
  }

  uint16_t handle = 0;
  if ((nrhs >= 2) && (!mxIsEmpty(prhs[1]))) {
    handle = get_handle(prhs[0]);
  }
  shared_ptr<Net<float> > net = get_net(handle);
  if (!net) {
    mex_error("Net was uninitialized in call to 'backward'");
  }

  plhs[0] = do_backward(prhs[0], net);
}

static void is_initialized(MEX_ARGS) {
  if ((nrhs != 0) && (nrhs != 1)) {
    mex_error("Wrong number of arguments. "
              "Usage: caffe('is_initialized'"
              "[, net_handle=<common>])");
  }

  uint16_t handle = 0;
  if ((nrhs >= 1) && (!mxIsEmpty(prhs[0]))) {
    handle = get_handle(prhs[0]);
  }
  shared_ptr<Net<float> > net = get_net(handle);

  if (!net) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
      mex_error("Wrong number of arguments. "
                "Usage: caffe('read_mean', "
                "'path_to_binary_mean_file')");
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mex_error("Couldn't read mean file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

static void get_backend(MEX_ARGS) {
    if ((nlhs != 1) || (nrhs != 0)) {
      mex_error("Wrong number of arguments. "
                "Usage: backend = caffe('get_backend')");
    }

    switch (Caffe::mode()) {
    case Caffe::CPU:
      plhs[0] = mxCreateString("cpu");
      break;
    case Caffe::GPU:
      plhs[0] = mxCreateString("gpu");
      break;
    default:
      mex_error("Could not recognize backend");
    }
}

static void get_blob_size(MEX_ARGS) {
    if (((nrhs != 1) && (nrhs != 2)) || (nlhs != 1)) {
      mex_error("Wrong number of arguments. "
                "Usage: size = caffe('get_blob_size', "
                "blob_names=<[]=final>[, net_handle=<common>])");
    }

    // Handle params
    std::vector<std::string> blob_names;
    uint16_t handle = 0;

    if ((nrhs >= 2) && (!mxIsEmpty(prhs[1]))) {
      handle = get_handle(prhs[1]);
    }

    shared_ptr<Net<float> > net = get_net(handle);
    if (!net) {
      mex_error("Net was uninitialized in call to 'get_blob_size'");
    }

    if ((nrhs >= 1) && (!mxIsEmpty(prhs[0]))) {
      if (!mxIsCell(prhs[0])) {
        mex_error("blob_names should be a cell array of strings");
      }
      size_t blob_count = mxGetNumberOfElements(prhs[0]);
      for (size_t i = 0; i < blob_count; ++i) {
        std::string blob_name =
          std::string(mxArrayToString(mxGetCell(prhs[0], i)));
        blob_names.push_back(blob_name);
      }
    } else {
      const vector<string>& all_blob_names = net->blob_names();
      blob_names.push_back(all_blob_names[all_blob_names.size()-1]);
    }

    // Get named blob size
    plhs[0] = mxCreateCellMatrix(blob_names.size(), 1);

    for (size_t i = 0; i < blob_names.size(); ++i) {
      const shared_ptr<Blob<float> > custom_output_blob =
        net->blob_by_name(blob_names[i]);
      mxArray* mx_blob_sz =
        mxCreateNumericMatrix(4, 1, mxDOUBLE_CLASS, mxREAL);
      double* data = static_cast<double*>(mxGetData(mx_blob_sz));
      data[0] = custom_output_blob->num();
      data[1] = custom_output_blob->channels();
      data[2] = custom_output_blob->height();
      data[3] = custom_output_blob->width();
      mxSetCell(plhs[0], i, mx_blob_sz);
    }
}

static void destroy_net_by_handle(MEX_ARGS) {
    if (nrhs != 1) {
      mex_error("Wrong number of arguments. "
                "Usage: caffe('destroy_net_by_handle', "
                "net_handle)");
    }

    remove_net(get_handle(prhs[0]));
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
  { "backward",           backward        },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  { "get_backend",        get_backend     },
  { "get_blob_size",      get_blob_size   },
  // Custom handle management
  { "destroy_net_by_handle", destroy_net_by_handle},
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mex_error("No API command given");
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
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}
