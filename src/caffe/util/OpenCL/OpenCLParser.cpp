#ifdef USE_OPENCL

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <glog/logging.h>
#include <fstream>
#include <caffe/util/OpenCL/OpenCLParser.hpp>
#include <boost/regex.hpp>

namespace caffe {

OpenCLParser::OpenCLParser() {
}

OpenCLParser::~OpenCLParser() {
}

bool OpenCLParser::getKernelNames(std::string fileName, std::vector<std::string>& names) {

	if ( access( fileName.c_str(), F_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << fileName.c_str() << "' doesn't exist";
		return false;
	}

	if ( access( fileName.c_str(), R_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << fileName.c_str() << "' isn't readable";
		return false;
	}

	std::ifstream file;
	file.open(fileName.c_str(), std::ifstream::in );
	if ( ! file.is_open() ) {
		LOG(ERROR) << "failed to open file = '" << fileName.c_str() << "' for reading";
		return false;
	}

	boost::regex kernel_line("^__kernel[[:space:]]+void[[:space:]]+[[:word:]]+\\(.*", boost::regex::perl);
	boost::regex template_kernel_line("^template\\s+<class \\w+>\\s+__kernel[[:space:]]+void[[:space:]]+[[:word:]]+\\(.*", boost::regex::perl);
	boost::regex split("[\\s+(]");

	boost::smatch what;
    std::string str;

    while (std::getline(file, str)) {
    	//LOG(ERROR)<<str.c_str();

    	if ( boost::regex_match(str, what, kernel_line, boost::match_default) ) {
    		for(int m = 0; m < what.size(); m++) {
    			//LOG(ERROR)<<"match";
    			//std::cout << "match[" << m << "] = \'" << what[m] << "\'" << std::endl;
    			boost::sregex_token_iterator it(str.begin(),str.end(), split, -1);
    			it++;
    			it++;
    			names.push_back(*it);
    		}
    	}

    	if ( boost::regex_match(str, what, template_kernel_line, boost::match_default) ) {
    		for(int m = 0; m < what.size(); m++) {
    			//std::cout << "match[" << m << "] = \'" << what[m] << "\'" << std::endl;
    			boost::sregex_token_iterator it(str.begin(),str.end(), split, -1);
    			it++;
     			it++;
    			it++;
    			it++;
    			it++;
    			names.push_back(*it+"Float");
    			names.push_back(*it+"Double");
    		}
    	}
    }
	return true;
}

bool OpenCLParser::match(std::string line, boost::regex re) {

	boost::smatch what;

	if ( ! boost::regex_match(line, what, re, boost::match_default) ) {
		return false;
	}
	return true;
}

bool OpenCLParser::isKernelLine(std::string line) {

	boost::regex re("^__kernel[[:space:]]+void[[:space:]]+[[:word:]]+\\(.*", boost::regex::perl);
	return this->match(line, re);
}

bool OpenCLParser::isTemplateKernelLine(std::string line) {

	boost::regex re("^template\\s+<class \\w+>\\s+__kernel[[:space:]]+void[[:space:]]+[[:word:]]+\\(.*", boost::regex::perl);
	return this->match(line, re);
}

bool OpenCLParser::isAttributeLine(std::string line) {

	boost::regex re("^template __attribute__\\(\\(mangled_name\\(.*", boost::regex::perl);
	return this->match(line, re);
}

std::string OpenCLParser::getKernelName(std::string line) {

	boost::regex split_bgn("\\_*\\_*kernel\\s+void\\s+");
	boost::regex split_end("\\(");
	boost::sregex_token_iterator it;

	it = boost::sregex_token_iterator(line.begin(),line.end(), split_bgn, -1);
	it++;
	line = (*it);
	it = boost::sregex_token_iterator(line.begin(),line.end(), split_end, -1);
	line = *it;
	return line;
}

std::string OpenCLParser::getKernelType(std::string line) {

	boost::regex split_bgn("template\\s+<class\\s+");
	boost::regex split_end("\\s*>");
	boost::sregex_token_iterator it;

	it = boost::sregex_token_iterator(line.begin(),line.end(), split_bgn, -1);
	it++;
	line = (*it);
	it = boost::sregex_token_iterator(line.begin(),line.end(), split_end, -1);
	line = *it;
	return line;
}

std::string OpenCLParser::getTypedKernelName(std::string line) {

	boost::regex split_bgn("template\\s+__attribute__\\(\\(mangled_name\\(");
	boost::regex split_end("\\)");
	boost::sregex_token_iterator it;

	it = boost::sregex_token_iterator(line.begin(),line.end(), split_bgn, -1);
	it++;
	line = (*it);
	it = boost::sregex_token_iterator(line.begin(),line.end(), split_end, -1);
	line = *it;
	return line;
}

std::string OpenCLParser::getTypedKernelLine(std::string line) {

	std::string kernel_name = getKernelName(line);
	boost::regex split_bgn(kernel_name);
	boost::regex split_end("\\;");
	boost::sregex_token_iterator it;

	it = boost::sregex_token_iterator(line.begin(),line.end(), split_bgn, -1);
	it++;
	it++;
	line = (*it);
	it = boost::sregex_token_iterator(line.begin(),line.end(), split_end, -1);
	line = *it;
	return line;
}

bool OpenCLParser::isFloatType(std::string name) {

	boost::regex re(".*Float", boost::regex::perl);
	return this->match(name, re);
}

bool OpenCLParser::isDoubleType(std::string name) {

	boost::regex re(".*Double", boost::regex::perl);
	return this->match(name, re);
}

bool OpenCLParser::convert(std::string fileNameIN, std::string fileNameOUT) {

	if ( access( fileNameIN.c_str(), F_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << fileNameIN.c_str() << "' doesn't exist";
		return false;
	}

	if ( access( fileNameIN.c_str(), R_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << fileNameIN.c_str() << "' isn't readable";
		return false;
	}

	if ( access( fileNameOUT.c_str(), F_OK ) == 0 ) {

		 struct stat statIN;
		 if (stat(fileNameIN.c_str(), &statIN) == -1) {
		       perror(fileNameIN.c_str());
		       return false;
		 }

		 struct stat statOUT;
		 if (stat(fileNameOUT.c_str(), &statOUT) == -1) {
		       perror(fileNameOUT.c_str());
		       return false;
		 }

		 if ( statOUT.st_mtime > statIN.st_mtime ) {
			 DLOG(INFO) << "kernel source file = '" << fileNameOUT.c_str() << "' up-to-date";
			 return true;
		 }
	}

	std::ifstream file;
	file.open(fileNameIN.c_str(), std::ifstream::in );
	if ( ! file.is_open() ) {
		LOG(ERROR) << "failed to open file = '" << fileNameIN.c_str() << "' for reading";
		return false;
	}

    std::string line;
    std::string kernel_buffer;
    std::string kernel_name;
    std::string kernel_type;
    std::string kernel_name_typed;
    std::string kernel_line_typed;
    std::string kernel_modified;
    std::string type_replace;
    std::string stdOpenCL;

    stdOpenCL += "// This file was auto-generated from file '" + fileNameIN + "' to conform to standard OpenCL\n";

    bool recording = false;

    while (std::getline(file, line)) {

    	if ( isAttributeLine(line) ) {
    		if ( recording ) {
          		recording = false;
       		}
        	kernel_name_typed = getTypedKernelName(line);
        	kernel_line_typed = "__kernel void " + kernel_name_typed + getTypedKernelLine(line) + " {";

        	if ( isFloatType(kernel_name_typed) ) {
        		type_replace = "float";
        	}
        	if ( isDoubleType(kernel_name_typed) ) {
        		type_replace = "double";
        	}

        	kernel_modified = kernel_line_typed + "\n" + kernel_buffer;

       		boost::regex re;
       		re = boost::regex("\\sT\\s", boost::regex::perl);
       		kernel_modified = boost::regex_replace(kernel_modified, re, " "+type_replace+" ");

       		re = boost::regex("\\sT\\*\\s", boost::regex::perl);
       		kernel_modified = boost::regex_replace(kernel_modified, re, " "+type_replace+"* ");

        	stdOpenCL += kernel_modified;
        	continue;
    	}

    	if ( isTemplateKernelLine(line) ) {
    		kernel_name = getKernelName(line);
    		kernel_type = getKernelType(line);
        DLOG(INFO)<<"found template kernel '"<<kernel_name<<"' with type '"<<kernel_type<<"'";

        	if ( recording == false ) {
        		recording = true;
        	} else {
        		LOG(ERROR) << "error parsing kernel source file = '" << fileNameIN.c_str() << "'";
        		return false;
        	}
        	continue;
    	}

    	if ( recording ) {
    		kernel_buffer += line + "\n";
    	} else {
        	kernel_buffer = "";
        	stdOpenCL += line + "\n";
    	}

    }

    std::ofstream out(fileNameOUT.c_str());
    out << stdOpenCL;
    out.close();
    DLOG(INFO) << "convert AMD OpenCL '"<<fileNameIN.c_str()<<"' to standard OpenCL '"<<fileNameOUT.c_str()<<"'";
	return true;
}


} // namespace caffe

#endif // USE_OPENCL
