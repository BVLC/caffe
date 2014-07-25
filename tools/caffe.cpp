#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::string;
using std::vector;


// Used in device query
DEFINE_int32(device_id, 0,
             "[devicequery,speedtest] The device id to use.");
// Used in training
DEFINE_string(solver_proto_file, "",
              "[train] The protobuf containing the solver definition.");
DEFINE_string(net_proto_file, "",
              "[speedtest] The net proto file to use.");
DEFINE_string(resume_point_file, "",
              "[train] (optional) The snapshot from which to resume training.");
DEFINE_string(pretrained_net_file, "",
              "[train] (optional) A pretrained network to run finetune from. "
              "Cannot be set simultaneously with resume_point_file.");
DEFINE_int32(run_iterations, 50,
             "[speedtest] The number of iterations to run.");
DEFINE_bool(speedtest_with_gpu, false,
            "[speedtest] Test the model with GPU.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
  namespace { \
  class __Registerer_##func { \
   public: \
    __Registerer_##func() { \
      g_brew_map[#func] = &func; \
    } \
  }; \
  __Registerer_##func g_registerer_##func; \
  }

static BrewFunction GetBrewFunction(const std::string& name) {
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

// caffe actions that could be called in the form
//     caffe.bin action
// To do so, define actions as "int action()" functions, and register it with
// RegisterBrewFunction(action);

int devicequery() {
  LOG(INFO) << "Querying device_id = " << FLAGS_device_id;
  caffe::Caffe::SetDevice(FLAGS_device_id);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(devicequery);

int train() {
  CHECK_GT(FLAGS_solver_proto_file.size(), 0);

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver_proto_file, &solver_param);

  LOG(INFO) << "Starting Optimization";
  caffe::SGDSolver<float> solver(solver_param);
  if (FLAGS_resume_point_file.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_resume_point_file;
    solver.Solve(FLAGS_resume_point_file);
  } else if (FLAGS_pretrained_net_file.size()) {
    LOG(INFO) << "Finetuning from " << FLAGS_pretrained_net_file;
    solver.net()->CopyTrainedLayersFrom(FLAGS_pretrained_net_file);
    solver.Solve();
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);

int speedtest() {
  // Set device id and mode
  if (FLAGS_speedtest_with_gpu) {
    LOG(INFO) << "Use GPU with device id " << FLAGS_device_id;
    Caffe::SetDevice(FLAGS_device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Caffe::set_phase(Caffe::TRAIN);
  Net<float> caffe_net(FLAGS_net_proto_file);

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
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_run_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  forward_timer.Start();
  Timer timer;
  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < FLAGS_run_iterations; ++j) {
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    LOG(INFO) << layername << "\tforward: " << timer.MilliSeconds() <<
        " milli seconds.";
  }
  LOG(INFO) << "Forward pass: " << forward_timer.MilliSeconds() <<
      " milli seconds.";
  Timer backward_timer;
  backward_timer.Start();
  for (int i = layers.size() - 1; i >= 0; --i) {
    const string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < FLAGS_run_iterations; ++j) {
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          &bottom_vecs[i]);
    }
    LOG(INFO) << layername << "\tbackward: "
        << timer.MilliSeconds() << " milli seconds.";
  }
  LOG(INFO) << "Backward pass: " << backward_timer.MilliSeconds() <<
      " milli seconds.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() <<
      " milli seconds.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(speedtest);

int main(int argc, char** argv) {
  caffe::GlobalInit(&argc, &argv);
  CHECK_EQ(argc, 2);
  return GetBrewFunction(std::string(argv[1]))();
}
