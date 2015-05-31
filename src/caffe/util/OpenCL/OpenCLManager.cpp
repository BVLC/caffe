#ifdef USE_OPENCL

#include <glog/logging.h>
#include <CL/cl.h>
#include <caffe/util/OpenCL/OpenCLManager.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>

namespace caffe {

OpenCLManager OpenCLManager::instance_;

OpenCLManager::OpenCLManager():
  initialized_(false) {
}

OpenCLManager::~OpenCLManager() {
}

bool OpenCLManager::Init() {
  if (instance_.initialized_) {
    return true;
  }
  LOG(INFO) << "Initialize OpenCL";
  instance_.Query();
  if ( OpenCLManager::GetNumPlatforms() <= 0 ) {
    LOG(FATAL) << "No OpenCL platforms found.";
    return false;
  }

  // TODO: mechanism for choosing the correct platform.
  instance_.current_platform_index_ = 0;

  OpenCLPlatform& pf = CurrentPlatform();
  pf.print();

  if (!pf.createContext()) {
    LOG(FATAL) << "failed to create OpenCL context for platform " << pf.name();
    return false;
  }

  std::vector<std::string> cl_files;
  cl_files.push_back("src/caffe/util/OpenCL/math_functions.cl");
  cl_files.push_back("src/caffe/util/OpenCL/im2col.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/pooling_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/relu_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/prelu_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/sigmoid_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/tanh_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/dropout_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/bnll_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/contrastive_loss_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/eltwise_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/lrn_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/softmax_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/softmax_loss_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/threshold_layer.cl");
  cl_files.push_back("src/caffe/layers/OpenCL/mvn_layer.cl");

  std::vector<std::string>::iterator it;

  for ( it = cl_files.begin(); it != cl_files.end(); it++ ) {
    if ( !pf.compile(*it) ) {
      LOG(FATAL) << "failed to create to create OpenCL program for platform " << pf.name();
      return false;
    }
  }

  if ( pf.getNumGPUDevices() < 1 ) {
    LOG(FATAL) << "No GPU devices available at platform " << pf.name();
    return false;
  }
  pf.SetCurrentDevice(CL_DEVICE_TYPE_GPU, instance_.device_id_);
  OpenCLDevice& device = pf.CurrentDevice();
  if (!device.createQueue()) {
    LOG(FATAL) << "failed to create OpenCL command queue for device "
               << device.name();
    return false;
  }

  if ( clblasSetup() != CL_SUCCESS ) {
    LOG(FATAL) << "failed to initialize clBlas";
    return false;
  }

  device.print();
  instance_.initialized_ = true;
  return true;
}

bool OpenCLManager::Query() {
  typedef std::vector<cl::Platform> ClPlatforms;
  typedef ClPlatforms::iterator ClPlatformsIter;

  ClPlatforms cl_platforms;
  cl::Platform::get(&cl_platforms);
  if (cl_platforms.empty()) {
    LOG(INFO) << "found no OpenCL platforms.";
    return false;
  }

  for(ClPlatformsIter it = cl_platforms.begin(); it != cl_platforms.end(); ++it) {
    OpenCLPlatform plat(*it);
    if (!plat.Query()) {
      LOG(ERROR) << "failed to query platform.";
      return false;
    }
    platforms_.push_back(plat);
  }
  LOG(INFO) << "found " << platforms_.size() << " OpenCL platforms";

  // FIXME: platform is the first one.
  current_platform_index_ = 0;

	return true;
}

void OpenCLManager::Print() {
	std::cout << "-- OpenCL Manager Information -----------------------------------" << std::endl;
  for (PlatformIter it = instance_.platforms_.begin();
       it != instance_.platforms_.end(); it++) {
    it->print();
	}
}

int OpenCLManager::GetNumPlatforms() {
  return instance_.platforms_.size();
}

OpenCLPlatform* OpenCLManager::getPlatform(unsigned int idx) {
  if ( idx >= platforms_.size() ) {
		LOG(ERROR) << "platform idx = " << idx << " out of range.";
		return NULL;
	}
  return &platforms_[idx];
}

OpenCLPlatform& OpenCLManager::CurrentPlatform() {
  if (instance_.current_platform_index_ < 0) {
    LOG(FATAL) << "No current platform.";
  }
  return instance_.platforms_[instance_.current_platform_index_];
}

void OpenCLManager::SetDeviceId(int device_id) {
  instance_.device_id_ = device_id;
}

} // namespace caffe

#endif // USE_OPENCL
