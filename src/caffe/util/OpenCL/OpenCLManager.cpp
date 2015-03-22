#ifdef USE_OPENCL

#include <glog/logging.h>
#include <CL/cl.h>
#include <caffe/util/OpenCL/OpenCLManager.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>

namespace caffe {

OpenCLManager::OpenCLManager() {

	numOpenCLPlatforms 	= 0;
	platformPtr			= NULL;
}

OpenCLManager::~OpenCLManager() {
	free(platformPtr);
}

bool OpenCLManager::query() {

	if ( ! CL_CHECK(clGetPlatformIDs(0, NULL, &numOpenCLPlatforms)) ) {
		return false;
	}
	LOG(INFO) << "found " << numOpenCLPlatforms << " OpenCL platforms";

	platformPtr = (cl_platform_id*) malloc(numOpenCLPlatforms * sizeof(cl_platform_id));
	if (platformPtr == NULL) {
		LOG(ERROR) << "failed to allocate memory";
		return false;
	}

	if ( ! CL_CHECK( clGetPlatformIDs(numOpenCLPlatforms, platformPtr, NULL) ) ) {
		return false;
	}

	for (int i = 0; i < numOpenCLPlatforms; i++) {

		OpenCLPlatform pf = OpenCLPlatform(platformPtr[i]);
		if ( ! pf.query() ) {
			LOG(ERROR) << "failed to query platform " << i;
			return false;
		}
		platforms.push_back(pf);
	}

	return true;
}



void OpenCLManager::print() {

	std::vector<caffe::OpenCLPlatform>::iterator it;
	std::cout << "-- OpenCL Manager Information -----------------------------------" << std::endl;
	for (it = platforms.begin(); it != platforms.end(); it++) {
		(*it).print();
	}
}

int OpenCLManager::getNumPlatforms() {

	return numOpenCLPlatforms;
}

OpenCLPlatform* OpenCLManager::getPlatform(unsigned int idx) {

	if ( idx >= platforms.size() ) {
		LOG(ERROR) << "platform idx = " << idx << " out of range.";
		return NULL;
	}
	return &platforms[idx];
}

} // namespace caffe

#endif // USE_OPENCL
