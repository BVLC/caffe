#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "boost/algorithm/string.hpp"
#include "boost/make_shared.hpp"
#include "caffe/caffe.hpp"
#include "caffe/multinode/multinode.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(model_ref, "",
              "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");
DEFINE_string(param_server, "",
    "Optional; multinode mode, "
    "the parent param server address to synchronize with, "
    "i.e.: tcp://127.0.0.1:7777");
DEFINE_string(listen_address, "",
    "Optional; multinode mode, bind address for various servers");
DEFINE_string(multinode_type, "sync",
    "Optional; multinode mode, type of multinode training mode "
    "[sync, async, ave]");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
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

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

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
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  if (caffe::internode::is_remote_address(FLAGS_solver)) {
    caffe::ReceiveProtoFromRemoteOrDie(FLAGS_solver, &solver_param);
  } else {
    LOG(INFO) << "assuming solver is local file";
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();

    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (FLAGS_param_server != "") {
    LOG(INFO) << "Configuring multinode setup";

    if (FLAGS_multinode_type.find("sync") == 0) {
      caffe::SynchronousParamClient<float> sync(solver, FLAGS_param_server);
      LOG(INFO) << "Starting Multi-node Optimization";
      sync.run();
    } else if (FLAGS_multinode_type.find("ave") == 0) {
      LOG(ERROR) << "currently unsupported";
      return 0;
    } else if (FLAGS_multinode_type.find("async") == 0) {
      LOG(ERROR) << "currently unsupported";
      return 0;
    } else {
      LOG(ERROR) << "Invalid multinode type " << FLAGS_param_server;
    }
  } else if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);

template <template<typename T> class ServerType>
int run_server(string name) {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }
  LOG(INFO) << "Starting " << name;
  ServerType<float> server(solver, FLAGS_listen_address);
  server.run();
  return 0;
}

int param_server() {
  if (FLAGS_multinode_type.find("sync") == 0) {
    return run_server<caffe::SynchronousParamServer>("Param Server");
  } else if (FLAGS_multinode_type.find("ave") == 0) {
    LOG(ERROR) << "currently unsupported";
  } else if (FLAGS_multinode_type.find("async") == 0) {
    LOG(ERROR) << "currently unsupported";
  } else {
    LOG(ERROR) << "Invalid multinode type " << FLAGS_param_server;
  }
  return 1;
}
RegisterBrewFunction(param_server);

int model_server() {
  return run_server<caffe::ModelServer>("Model Server");
}
RegisterBrewFunction(model_server);

int data_server() {
  return run_server<caffe::DataServer>("Data Server");
}
RegisterBrewFunction(data_server);

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<float> forward_time_per_layer(layers.size(), 1e37);
  std::vector<float> backward_time_per_layer(layers.size(), 1e37);
  float forward_time = 0;
  float backward_time = 0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] = std::min(forward_time_per_layer[i],timer.MicroSeconds());
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] = std::min(backward_time_per_layer[i], timer.MicroSeconds());
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Min time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward:  " << forward_time_per_layer[i] / 1000 /
      (1 + 0*FLAGS_iterations) << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      (1+0*FLAGS_iterations) << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

// compare: functional testing against another implementation
int compare() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to compare.";
  CHECK_GT(FLAGS_model_ref.size(), 0) << "Need a model_ref definition to compare.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net    (FLAGS_model, caffe::TRAIN);
  Net<float> caffe_net_ref(FLAGS_model_ref, caffe::TRAIN);

  float initial_loss;

  // Initial fw/bw pass - all the allocations and initializations will be done
  LOG(INFO) << "\n\nPerforming Forward - REF";
  caffe_net_ref.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward - REF";
  caffe_net_ref.Backward();

  LOG(INFO) << "\n\nPerforming Forward";
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  const vector<shared_ptr<Blob<float> > >& params = caffe_net.params();

  const vector<shared_ptr<Layer<float> > >& layers_ref = caffe_net_ref.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs_ref = caffe_net_ref.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs_ref    = caffe_net_ref.top_vecs();
  const vector<vector<bool> >& bottom_need_backward_ref =
    caffe_net_ref.bottom_need_backward();
  const vector<shared_ptr<Blob<float> > >& params_ref      = caffe_net_ref.params();

  CHECK_EQ(layers.size()      , layers_ref.size());
  CHECK_EQ(bottom_vecs.size() , bottom_vecs_ref.size());
  CHECK_EQ(top_vecs.size()    , top_vecs_ref.size());
  CHECK_EQ(params.size()      , params_ref.size());

  for (int param_id = 0; param_id < params_ref.size();
       ++param_id)
  {
    // use the same initial parameters values
    caffe::caffe_copy(
               params_ref[param_id]->count(),
               params_ref[param_id]->cpu_data(),
               params[param_id]->mutable_cpu_data());

    if (params[param_id]->prv_diff())
      caffe::caffe_set(
        params[param_id]->prv_diff_count(),
        0.f,
        params[param_id]->mutable_prv_diff());
    else
      caffe::caffe_set(
        params[param_id]->count(),
        0.f,
        params[param_id]->mutable_cpu_diff());

    if (params_ref[param_id]->prv_diff())
      caffe::caffe_set(
        params_ref[param_id]->prv_diff_count(),
        0.f,
        params_ref[param_id]->mutable_prv_diff());
    else
      caffe::caffe_set(
        params_ref[param_id]->count(),
        0.f,
        params_ref[param_id]->mutable_cpu_diff());

  }

  // Data layer from reference:
  layers_ref[0]->Forward(bottom_vecs_ref[0], top_vecs_ref[0]);
  // Reuse data layer output from reference net
  const_cast< vector<vector<Blob<float>*> >& > (bottom_vecs)[1] = bottom_vecs_ref[1];

  LOG(INFO) << "\n\nPerforming Forward - collect data for comparison";
  // start after the data layer
  for (int i = 1; i < layers.size(); ++i) {

    CHECK_EQ(layers[i]->layer_param().name(), layers_ref[i]->layer_param().name());

    layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    layers_ref[i]->Forward(bottom_vecs_ref[i], top_vecs_ref[i]);
  }

  LOG(INFO) << "\n\nCompare fwd output, layer by layer";
  // start after the data layer
  for (int i = 1; i < layers.size(); ++i) {

    CHECK_EQ(layers[i]->layer_param().name(), layers_ref[i]->layer_param().name());

    // compare vs reference
    const float max_allowed_error = 0.05; // 1%
    const float very_small_value = 0.0001; // ;-)
    const float* data =    top_vecs[i][0]->cpu_data();
    const float* ref = top_vecs_ref[i][0]->cpu_data();

    CHECK_EQ(top_vecs_ref[i][0]->count(), top_vecs[i][0]->count());
    for (int j = 0; j < top_vecs_ref[i][0]->count(); ++j) {
      float err = (fabs(ref[j]) < very_small_value)  ? (fabs(data[j] - ref[j])) : (fabs( (data[j] - ref[j]) / ref[j]));
      if (err > max_allowed_error) LOG(INFO) <<
                                  "Forward: Error " << err << " at offset " << j <<  " vals: " << data[j] << " should be " << ref[j]
                                  << " layer: " << i << " name: " << layers[i]->layer_param().name();
    }
  }

  LOG(INFO) << "\n\nPerforming Backward - collect data for comparison";
  // Loss layer (we need loss based on the same labels)
  int loss_id = layers.size() -1;
  layers_ref[loss_id]->Backward(top_vecs_ref[loss_id], bottom_need_backward_ref[loss_id],
                          bottom_vecs_ref[loss_id]);

  // Reuse loss layer output from reference net
  const_cast< vector<vector<Blob<float>*> >& > (top_vecs)[loss_id-1] = top_vecs_ref[loss_id-1];
  // Start after loss layer
  for (int i = loss_id - 1; i > 0; --i) {
    CHECK_EQ(layers[i]->layer_param().name(), layers_ref[i]->layer_param().name());
    layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                        bottom_vecs[i]);
    layers_ref[i]->Backward(top_vecs_ref[i], bottom_need_backward_ref[i],
                            bottom_vecs_ref[i]);
  }

  LOG(INFO) << "\n\nVeryfing Backward - compare bwd output, layer by layer";
  for (int i = loss_id - 1; i > 0; --i) {
    // compare vs reference
    const float max_allowed_error = 0.01; // 1%
    const float very_small_value = 0.0001; // ;-)
    const float* data =    bottom_vecs[i][0]->cpu_diff();
    const float* ref = bottom_vecs_ref[i][0]->cpu_diff();

    CHECK_EQ(bottom_vecs_ref[i][0]->count(), bottom_vecs[i][0]->count());
    for (int j = 0; j < bottom_vecs_ref[i][0]->count(); ++j) {
      float err = (fabs(ref[j]) < very_small_value) ? (fabs(data[j] - ref[j])) : (fabs((data[j] - ref[j]) / ref[j]));
      if (err > max_allowed_error) LOG(INFO) <<
         "Backward: Error " << err << " at offset " << j <<  " vals: " << data[j] << " should be " << ref[j]
         << " layer: " << i << " name: " << layers[i]->layer_param().name();
    }
  }

  LOG(INFO) << "\n\nChecking parameters gradients";
  // compare parameters gradients
  for (int param_id = params_ref.size() - 1; param_id >= 0;
       --param_id) {
  //for (int param_id = 0 ; param_id < params_ref.size();
  //     ++param_id) {

    const float max_allowed_error = .1; // 10%
    const float very_small_value = 0.005; // ;-)
    CHECK_EQ(params[param_id]->count(), params_ref[param_id]->count());

    const float* data =    params[param_id]->cpu_diff();
    const float* ref = params_ref[param_id]->cpu_diff();
    bool has_err = false;

    for (int j = 0; j < params_ref[param_id]->count(); ++j) {
      float err = (fabs(ref[j]) < very_small_value)  ? (fabs(data[j] - ref[j])) : (fabs( (data[j] - ref[j]) / ref[j]));
      if (err > max_allowed_error) {
        LOG(INFO) <<
          " Param diff: Error " << err << " at offset " << j <<  " vals: " << data[j] << " should be " << ref[j]
          << "   param_id: " << param_id ;
        has_err = true;
      }
    }
    if(has_err) break;
   // {
   //   //
   //   int num      = params[param_id]->num();
   //   int channels = params[param_id]->channels();
   //   int height   = params[param_id]->height();
   //   int width    = params[param_id]->width();
   //
   //   LOG(INFO) << " Dimensions of params at param_id: " << param_id << " - " << num << " " << channels << " " << height << " " << width;
   //
   //   for (int n = 0; n < num; n++)
   //     for (int c = 0; c < channels;  c++)
   //       for (int h = 0; h < height;  h++)
   //         for (int w = 0; w < width; w++)
   //           {
   //             float r = params_ref[param_id]->diff_at(n,c,h,w);
   //             float d = params[param_id]->diff_at(n,c,h,w);
   //
   //             float err = ( r == 0 || (fabs(r) < very_small_value) ) ? (fabs(d - r)) : (fabs( (d - r) / r ) );
   //             if (err > max_allowed_error) {
   //               has_err = true;
   //               LOG(INFO) <<
   //                 " Param diff: Error " << err << " at " << n <<  " " << c << " " << h << " " <<  w << " vals: " << d << " should be " << r;
   //               CHECK(0);
   //             }
   //           }
   // }

  }


  LOG(INFO) << "The End";
  return 0;
}
RegisterBrewFunction(compare);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  param_server    run param server - weights synchronizing entity\n"
      "  model_server    run model server - remote model source\n"
      "  data_server     run data server - remote data source\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
