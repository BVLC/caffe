// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.


#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)



// for convenience, check that input files can be opened, and raise an
// exception
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static int init_key = -2;

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
// 
// 
// Functions to copy data from mxarray to blobs and viceversa
// 

static mxArray* blob_data_to_mxarray(const Blob<float>* blob) {
  mwSize dims[4] = {blob->width(), blob->height(),
                    blob->channels(), blob->num()};
  DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
  mxArray* mx_blob_data = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob->count(), blob->cpu_data(), blob_data_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(blob->count(), blob->gpu_data(), blob_data_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return mx_blob_data;
}

static void mxarray_to_blob_data(const mxArray* data, Blob<float>* blob) {
  mwSize dims[4] = {blob->width(), blob->height(),
                    blob->channels(), blob->num()};
  DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
  CHECK_EQ_MEX(blob->count(),mxGetNumberOfElements(data),
    "blob->count() don't match with numel(data)");
  const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(data));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_cpu_data());
    break;
  case Caffe::GPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_gpu_data());
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())
}

static mxArray* blob_diff_to_mxarray(const Blob<float>* blob) {
  mwSize dims[4] = {blob->width(), blob->height(),
                    blob->channels(), blob->num()};
  DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
  mxArray* mx_blob_data = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob->count(), blob->cpu_diff(), blob_data_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(blob->count(), blob->gpu_diff(), blob_data_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return mx_blob_data;
}

static void mxarray_to_blob_diff(const mxArray* data, Blob<float>* blob) {
  mwSize dims[4] = {blob->width(), blob->height(),
                    blob->channels(), blob->num()};
  DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
  CHECK_EQ_MEX(blob->count(),mxGetNumberOfElements(data),
    "blob->count() don't match with numel(data)");
  const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(data));
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_cpu_diff());
    break;
  case Caffe::GPU:
    caffe_copy(blob->count(), data_ptr, blob->mutable_gpu_diff());
    break;
  default:
    LOG(FATAL) << "Unknown Caffe mode.";
  }  // switch (Caffe::mode())
}

static mxArray* blobs_data_to_cell(const vector<Blob<float>* >& blobs) {
  mxArray* mx_out = mxCreateCellMatrix(blobs.size(), 1);
  for (unsigned int i = 0; i < blobs.size(); ++i) {
    mxArray* mx_blob =  blob_data_to_mxarray(blobs[i]);
    mxSetCell(mx_out, i, mx_blob);
  }
  return mx_out;
}

static mxArray* blobs_diff_to_cell(const vector<Blob<float>* >& blobs) {
  mxArray* mx_out = mxCreateCellMatrix(blobs.size(), 1);
  for (unsigned int i = 0; i < blobs.size(); ++i) {
    mxArray* mx_blob =  blob_diff_to_mxarray(blobs[i]);
    mxSetCell(mx_out, i, mx_blob);
  }
  return mx_out;
}

// TODO(matcaffe) no loss pointer arg
static mxArray* do_forward_prefilled(mxArray* mx_loss) {
  float* loss_ptr = reinterpret_cast<float*>(mxGetPr(mx_loss));
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled(loss_ptr);
  DLOG(INFO) << "loss: " << mxGetScalar(mx_loss);
  mxArray* mx_out = blobs_data_to_cell(output_blobs);
  // mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  // for (unsigned int i = 0; i < output_blobs.size(); ++i) {
  //   mxArray* mx_blob =  blob_data_to_mxarray(output_blobs[i]);
  //   mxSetCell(mx_out, i, mx_blob);
  // }
  return mx_out;
}

// TODO(matcaffe) mex_error
// TODO(matcaffe) caffe_copy
static void do_set_input_blobs(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ_MEX(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  // Copy values into input_blobs
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    mxarray_to_blob_data(elem, input_blobs[i]);
  }
}

// TODO(matcaffe) no loss pointer arg
static mxArray* do_forward(const mxArray* const bottom, mxArray* mx_loss) {
  do_set_input_blobs(bottom);
  return do_forward_prefilled(mx_loss);
}

static mxArray* do_backward_prefilled() {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  DLOG(INFO) << "Start";
  net_->Backward();
  DLOG(INFO) << "End";
  mxArray* mx_out = blobs_diff_to_cell(input_blobs);
  return mx_out;
}

// TODO(matcaffe) see do_set_input_blobs
static void do_set_output_blobs(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  CHECK_EQ_MEX(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // Copy top_diff to the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    mxarray_to_blob_diff(elem, output_blobs[i]);
  }
}

static mxArray* do_backward(const mxArray* const top_diff) {
  do_set_output_blobs(top_diff);
  return do_backward_prefilled();
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

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
    const char* fnames[2] = {"layer_name", "weights"};
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

      mxArray* mx_layer_weights = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        // Copy weights
        mx_layer_weights = blobs_data_to_cell(layer_blobs);
        // Create Struct
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_weights);
        mxSetField(mx_layers, mx_layer_index, "layer_name",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;        
      }
    }
  }

  return mx_layers;
}

static mxArray* do_get_layer_weights(const mxArray* const layer_name) {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  char* c_layer_name = mxArrayToString(layer_name);
  DLOG(INFO) << c_layer_name;
  mxArray* mx_layer_weights = NULL;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    DLOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(),c_layer_name) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      mx_layer_weights = blobs_data_to_cell(layer_blobs);
      // const mwSize dims[2] = {layer_blobs.size(), 1};
      // mx_layer_weights = mxCreateCellArray(2, dims);
      // DLOG(INFO) << "layer_blobs.size()" << layer_blobs.size();
      // for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
      //   // internally data is stored as (width, height, channels, num)
      //   // where width is the fastest dimension
      //   mxArray* mx_weights = blob_data_to_mxarray(layer_blobs[j]);
      //   mxSetCell(mx_layer_weights, j, mx_weights);
      // }
    }
  }
  return mx_layer_weights;
}

static void do_set_layer_weights(const mxArray* const layer_name,
    const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  char* c_layer_name = mxArrayToString(layer_name);
  DLOG(INFO) << "Looking for: " << c_layer_name;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    DLOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(),c_layer_name) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      DLOG(INFO) << "Found layer " << layer_names[i];
      CHECK_EQ_MEX(static_cast<unsigned int>(mxGetDimensions(mx_layer_weights)[0]),
        layer_blobs.size(), "Num of cells don't match layer_blobs.size");
      DLOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j);
        mxarray_to_blob_data(elem, layer_blobs[j]);
        }
      }
    }
  }
}

static mxArray* do_get_layers_info() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  const int num_layers[2] = {layers.size(), 1};

  // Step 1: prepare output array of structures
  mxArray* mx_layers;
  {
    const char* fnames[3] = {"name", "type", "weights"};
    mx_layers = mxCreateStructArray(2, num_layers, 3, fnames);
  }

  // Step 2: copy info into output
  {
    mxArray* mx_blob;
    const char* blobfnames[5] = {"num", "channels", "height", "width", "count"};
    for (unsigned int i = 0; i < layers.size(); ++i) {
      mxSetField(mx_layers, i, "name",
        mxCreateString(layer_names[i].c_str()));
      mxSetField(mx_layers, i, "type",
        mxCreateString(layers[i]->type_name().c_str()));

      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      const int num_blobs[1] = {layer_blobs.size()};
      mx_blob = mxCreateStructArray(1, num_blobs, 5, blobfnames);

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        mxSetField(mx_blob, j, "num",
          mxCreateDoubleScalar(layer_blobs[j]->num()));
        mxSetField(mx_blob, j, "channels",
          mxCreateDoubleScalar(layer_blobs[j]->channels()));
        mxSetField(mx_blob, j, "height",
          mxCreateDoubleScalar(layer_blobs[j]->height()));
        mxSetField(mx_blob, j, "width",
          mxCreateDoubleScalar(layer_blobs[j]->width()));
        mxSetField(mx_blob, j, "count",
          mxCreateDoubleScalar(layer_blobs[j]->count()));
      }
      mxSetField(mx_layers, i, "weights", mx_blob);
    }
  }

  return mx_layers;
}

static mxArray* do_get_blobs_info() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_blobs;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[6] = {"name", "num", "channels", "height", "width", "count"};
    mx_blobs = mxCreateStructArray(1, num_blobs, 6, fnames);
  }

  // Step 2: copy info into output
  {
    for (unsigned int i = 0; i < blobs.size(); ++i) {
      mxSetField(mx_blobs, i, "name",
        mxCreateString(blob_names[i].c_str()));
      mxSetField(mx_blobs, i, "num",
        mxCreateDoubleScalar(blobs[i]->num()));
      mxSetField(mx_blobs, i, "channels",
        mxCreateDoubleScalar(blobs[i]->channels()));
      mxSetField(mx_blobs, i, "height",
        mxCreateDoubleScalar(blobs[i]->height()));
      mxSetField(mx_blobs, i, "width",
        mxCreateDoubleScalar(blobs[i]->width()));
      mxSetField(mx_blobs, i, "count",
        mxCreateDoubleScalar(blobs[i]->count()));
    }
  }
  return mx_blobs;
}



static mxArray* do_get_blob_data(const mxArray* const blob_name) {
  // const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  // const vector<string>& blob_names = net_->blob_names();

  char* c_blob_name = mxArrayToString(blob_name);
  DLOG(INFO) << "Looking for: " << c_blob_name;

  mxArray* mx_blob_data = NULL;
  if (net_->has_blob(c_blob_name)) {
   mx_blob_data = blob_data_to_mxarray(net_->blob_by_name(c_blob_name).get()); 
  }

  // for (unsigned int i = 0; i < blobs.size(); ++i) {
  //   DLOG(INFO) << blob_names[i];
  //   if (strcmp(blob_names[i].c_str(), c_blob_name) == 0) {
  //     mx_blob_data = blob_data_to_mxarray(blobs[i]);
  //   }
  // }

  return mx_blob_data;
}

static mxArray* do_get_blob_diff(const mxArray* const blob_name) {
  // const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  // const vector<string>& blob_names = net_->blob_names();

  char* c_blob_name = mxArrayToString(blob_name);
  DLOG(INFO) << "Looking for: " << c_blob_name;

  mxArray* mx_blob_diff = NULL;
  if (net_->has_blob(c_blob_name)) {
   mx_blob_diff = blob_diff_to_mxarray(net_->blob_by_name(c_blob_name).get()); 
  }

  // for (unsigned int i = 0; i < blobs.size(); ++i) {
  //   DLOG(INFO) << blob_names[i];
  //   if (strcmp(blob_names[i].c_str(),c_blob_name) == 0) {
  //     mx_blob_diff = blob_diff_to_mxarray(blobs[i]);
  //   }
  // }

  return mx_blob_diff;
}

static mxArray* do_get_all_data() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_data;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "data"};
    mx_all_data = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_data = blob_data_to_mxarray(blobs[i]);
    mxSetField(mx_all_data, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_data, i, "data",mx_blob_data);
  }
  return mx_all_data;
}

static mxArray* do_get_all_diff() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_diff;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "diff"};
    mx_all_diff = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_diff = blob_diff_to_mxarray(blobs[i]);
    mxSetField(mx_all_diff, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_diff, i, "diff",mx_blob_diff);
  }
  return mx_all_diff;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void get_layer_weights(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_layer_weights(prhs[0]);
}

static void set_weights(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Given " << nrhs << " arguments expecting 1";
    mexErrMsgTxt("Wrong number of arguments");
  }
  const mxArray* const mx_weights = prhs[0];
  if (!mxIsStruct(mx_weights)) {
    mexErrMsgTxt("Input needs to be struct");
  }
  int num_layers = mxGetNumberOfElements(mx_weights);
  for (int i = 0; i < num_layers; ++i) {
    const mxArray* layer_name= mxGetField(mx_weights,i,"layer_name");
    const mxArray* weights= mxGetField(mx_weights,i,"weights");
    do_set_layer_weights(layer_name,weights);
  }
}

static void set_layer_weights(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments you need layer_name and cell of weights");
  }
  do_set_layer_weights(prhs[0],prhs[1]);
}

static void get_layers_info(MEX_ARGS) {
  plhs[0] = do_get_layers_info();
}

static void get_blobs_info(MEX_ARGS) {
  plhs[0] = do_get_blobs_info();
}

static void get_blob_data(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_blob_data(prhs[0]);
}

static void get_blob_diff(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_blob_diff(prhs[0]);
}

static void get_all_data(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_data();
}

static void get_all_diff(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_diff();
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
  if (nrhs != 3) {
    ostringstream error_msg;
    error_msg << "Expected 3 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);
  char* phase_name = mxArrayToString(prhs[2]);

  CheckFile(string(param_file));
  CheckFile(string(model_file));

  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mex_error("Unknown phase.");
  }

  net_.reset(new Net<float>(string(param_file), phase));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);
  mxFree(phase_name);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void init_net(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* phase_name = mxArrayToString(prhs[2]);

  CheckFile(string(param_file));

  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mex_error("Unknown phase.");
  }

  net_.reset(new Net<float>(string(param_file)));

  mxFree(param_file);
  mxFree(phase_name);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void load_net(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (net_) {
    char* model_file = mxArrayToString(prhs[0]);

    CheckFile(string(model_file));
    net_->CopyTrainedLayersFrom(string(model_file));

    mxFree(model_file);
    init_key = random();  // NOLINT(caffe/random_fn)
    if (nlhs == 1) {
      plhs[0] = mxCreateDoubleScalar(init_key);
    }
  } else {
    mexErrMsgTxt("Need to initialize the network first with init_net");
  }
}

static void save_net(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  if (net_) {
  char* model_file = mxArrayToString(prhs[0]);
  NetParameter net_param;
  net_->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, model_file);
  CheckFile(string(model_file));
  mxFree(model_file);
  } else {
    mexErrMsgTxt("Need to have a network to save");
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
  if (nrhs == 0) {
    //Forward without arguments behaves as forward_prefilled
    plhs[0] = do_forward_prefilled(plhs[1]);
  } else {
    plhs[0] = do_forward(prhs[0],plhs[1]);
  }
}

static void forward_prefilled(MEX_ARGS) {
  if (nrhs != 0) {
    mex_error("forward_prefilled takes no arguments");
  }
  plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
  plhs[0] = do_forward_prefilled(plhs[1]);

}

static void set_input_blobs(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Given " << nrhs << " arguments";
    mexErrMsgTxt("It takes one argument");
  }
  do_set_input_blobs(prhs[0]);
}


static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    mex_error("Too many input arguments.");
  }
  if (nrhs == 0) {
    //Backward without arguments behaves as backward_prefilled
    plhs[0] = do_backward_prefilled();
  } else {
    plhs[0] = do_backward(prhs[0]);
  }
}

static void backward_prefilled(MEX_ARGS) {
  if (nrhs != 0) {
    mex_error("backward_prefilled takes no arguments");
  }

  plhs[0] = do_backward_prefilled();
}

static void set_output_blobs(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Given " << nrhs << " arguments";
    mexErrMsgTxt("It takes one argument");
  }
  do_set_output_blobs(prhs[0]);
}


static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
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

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward           },
  { "backward",           backward          },
  { "forward_prefilled",  forward_prefilled },
  { "backward_prefilled", backward_prefilled},
  { "set_input_blobs",    set_input_blobs   },
  { "set_output_blobs",   set_output_blobs  },
  { "init",               init              },
  { "init_net",           init_net          },
  { "load_net",           load_net          },
  { "save_net",           save_net          },
  { "is_initialized",     is_initialized    },
  { "set_mode_cpu",       set_mode_cpu      },
  { "set_mode_gpu",       set_mode_gpu      },
  { "set_device",         set_device        },
  { "get_weights",        get_weights       },
  { "set_weights",        set_weights       },
  { "get_layer_weights",  get_layer_weights },
  { "set_layer_weights",  set_layer_weights },
  { "get_layers_info",    get_layers_info   },
  { "get_blobs_info",     get_blobs_info    },
  { "get_blob_data",      get_blob_data     },
  { "get_blob_diff",      get_blob_diff     },
  { "get_all_data",       get_all_data      },
  { "get_all_diff",       get_all_diff      },
  { "get_init_key",       get_init_key      },
  { "reset",              reset             },
  { "read_mean",          read_mean         },
  // The end.
  { "END",                NULL              },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
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
