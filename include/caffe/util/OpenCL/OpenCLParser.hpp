#ifndef __OPENCL_PARSER_HPP__
#define __OPENCL_PARSER_HPP__

#include <CL/cl.h>
#include <string>
#include <iostream>
#include <boost/regex.hpp>

namespace caffe {

class OpenCLParser {

public:
	OpenCLParser();
	~OpenCLParser();

	bool getKernelNames(std::string fileName, std::vector<std::string>& names);
	bool convert(std::string fileNameIN, std::string fileNameOUT);

protected:

	bool match(std::string line, boost::regex re);
	bool isKernelLine(std::string line);
	bool isTemplateKernelLine(std::string line);
	bool isAttributeLine(std::string line);
	std::string getKernelName(std::string line);
	std::string getKernelType(std::string line);
	std::string getTypedKernelName(std::string line);
	std::string getTypedKernelLine(std::string line);
	bool isFloatType(std::string name);
	bool isDoubleType(std::string name);

private:
};

} // namespace caffe

#endif // __OPENCL_PARSER_HPP__
