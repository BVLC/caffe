#ifndef __OPENCL_MANAGER_HPP__
#define __OPENCL_MANAGER_HPP__

#include <CL/cl.h>
#include <vector>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>

namespace caffe {

class OpenCLManager {

public:
	OpenCLManager();
	~OpenCLManager();

	bool query();
	void print();
	int getNumPlatforms();
	OpenCLPlatform* getPlatform(unsigned int idx);

protected:

private:

	unsigned int						numOpenCLPlatforms;
	cl_platform_id* 					platformPtr;
	std::vector<caffe::OpenCLPlatform> 	platforms;
};

}

#endif // __OPENCL_MANAGER_HPP__
