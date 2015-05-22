/**
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
 */

#ifndef LLOGGING_H
#define LLOGGING_H

//#ifdef HAVE_GLOG // HAVE_LIB_GLOG
#include <glog/logging.h>
//#else
#include <iostream>

// avoid fatal checks from glog
#define CAFFE_THROW_ON_ERROR

#ifdef CAFFE_THROW_ON_ERROR
#undef LOG
#undef LOG_IF
#undef CHECK
#undef CHECK_OP_LOG
#ifdef DEBUG
#undef CHECK_EQ
#endif
#include <sstream>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
		 ( std::ostringstream() << std::dec << x ) ).str()
class CaffeErrorException : public std::exception
{
public:
  CaffeErrorException(const std::string &s):_s(s) {}
  ~CaffeErrorException() throw() {}
  const char* what() const throw() { return _s.c_str(); }
  std::string _s;
};

#define CHECK(condition)						\
  if (GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition)))			\
    throw CaffeErrorException(std::string(__FILE__) + ":" + SSTR(__LINE__) + " / Check failed (custom): " #condition ""); \
  LOG_IF(ERROR, false) \
  << "Check failed (custom): " #condition " "

#define CHECK_OP_LOG(name, op, val1, val2, log) CHECK((val1) op (val2))
#ifdef DEBUG
#define CHECK_EQ(val1,val2) if (0) std::cerr
#endif
#endif

static std::string INFO="INFO";
static std::string WARNING="WARNING";
static std::string ERROR="ERROR";
static std::string FATAL="FATAL";

static std::ostream nullstream(0);

inline std::ostream& LOG(const std::string &severity,std::ostream &out=std::cout)
{
  if (severity != FATAL)
    {
      out << std::endl;
      out << severity << " - ";
      return out;
    }
  else
    {
      throw CaffeErrorException(std::string(__FILE__) + ":" + SSTR(__LINE__) + " / Fatal Caffe error"); // XXX: cannot report the exact location of the trigger...
    }
  //return out;
}

inline std::ostream& LOG_IF(const std::string &severity,const bool &condition,std::ostream &out=std::cout)
{
  if (condition)
    return LOG(severity,out);
  else return nullstream;
}

//#endif
#endif
