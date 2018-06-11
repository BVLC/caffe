/**
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
 */

#ifndef LLOGGING_H
#define LLOGGING_H

#include <spdlog/spdlog.h>
#include <boost/algorithm/string.hpp>
#include <iostream>

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm *pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value);  // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d",
             pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
    return buffer_;
  }

 private:
  char buffer_[9];
};

// avoid fatal checks from glog
#define CAFFE_THROW_ON_ERROR

// make sure we erase definitions by glog if any
#undef LOG
#undef LOG_IF
#undef CHECK
#undef CHECK_OP_LOG
#undef CHECK_EQ
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_OP_LOG
#undef CHECK_NOTNULL
#undef DCHECK
#undef DCHECK_LT
#undef DCHECK_GT
#undef DCHECK_LE
#undef DCHECK_GE
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DLOG
#undef DFATAL
#undef LOG_DFATAL
#undef LOG_EVERY_N

#ifdef CAFFE_THROW_ON_ERROR
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

static std::string INFO="INFO";
static std::string WARNING="WARNING";
static std::string ERROR="ERROR";
static std::string FATAL="FATAL";

#define GLOG_NO_ABBREVIATED_SEVERITIES

#define INFO INFO
#define WARNING WARNING
#define ERROR ERROR
#define FATAL FATAL

static std::ostream nullstream(0);

#define CHECK(condition)						\
  if (!(condition)) \
    throw CaffeErrorException(std::string(__FILE__) + ":" + SSTR(__LINE__) + " / Check failed (custom): " #condition ""); \
  nullstream									\
  << "Check failed (custom): " #condition " "

#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))

#define CHECK_OP_LOG(name, op, val1, val2, log) CHECK((val1) op (val2))
/* #ifdef DEBUG */
/* #define CHECK_EQ(val1,val2) if (0) std::cerr */
/* #endif */
#endif

#define CHECK_NOTNULL(x) \
  ((x) == NULL ? LOG(FATAL) << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)

#ifdef NDEBUG
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

class CaffeLogger
{
 public:
  CaffeLogger(const std::string &severity)
    :_severity(severity)
  {
    _console = spdlog::get("caffe");
    if (!_console)
#ifdef USE_SYSLOG
      _console = spdlog::syslog_logger("caffe");
#else
      _console = spdlog::stdout_logger_mt("caffe");
#endif
  }

  ~CaffeLogger()
    {
      if (_severity == "none" || _str.empty()) // ignore
	{}
      else if (_severity == INFO)
	_console->info(_str);
      else if (_severity == WARNING)
	_console->warn(_str);
      else if (_severity == ERROR)
	_console->error(_str);
    }
  
  friend CaffeLogger& operator<<(const CaffeLogger &cl, const std::string &rstr)
  {
    std::string str = rstr;
    const_cast<CaffeLogger&>(cl)._str += str;
    return const_cast<CaffeLogger&>(cl);
  }

  friend CaffeLogger& operator<<(const CaffeLogger &cl, const double &d)
  {
    std::string str = std::to_string(d);
    boost::trim_right_if(str,boost::is_any_of("\n"));
    const_cast<CaffeLogger&>(cl)._str += str;
    return const_cast<CaffeLogger&>(cl);
  }

  friend CaffeLogger& operator<<(const CaffeLogger &cl, const std::ostream &out)
  {
    std::stringstream sstr;
    sstr << out.rdbuf();
    const_cast<CaffeLogger&>(cl)._str += sstr.str();
    return const_cast<CaffeLogger&>(cl);
  }
  
  std::string _severity = INFO;
  std::shared_ptr<spdlog::logger> _console;
  std::string _str;
};

inline CaffeLogger LOG(const std::string &severity)
{
  if (severity != FATAL)
  {
    return CaffeLogger(severity);
  }
  else
    {
      throw CaffeErrorException(std::string(__FILE__) + ":" + SSTR(__LINE__) + " / Fatal Caffe error"); // XXX: cannot report the exact location of the trigger...
    }
}

inline CaffeLogger LOG_IF(const std::string &severity,const bool &condition)
{
  if (condition)
    return LOG(severity);
  else return CaffeLogger("none");
}

#ifdef NDEBUG
inline CaffeLogger DFATAL(const std::string &severity)
{
  (void)severity;
  return CaffeLogger("none");
}
inline CaffeLogger LOG_DFATAL(const std::string &severity)
{
  (void)severity;
  return CaffeLogger("none");
}
inline CaffeLogger DLOG(const std::string &severity)
{
  (void)severity;
  return CaffeLogger("none");
}
#else
inline CaffeLogger DFATAL(const std::string &severity)
{
  (void)severity;
  return LOG(FATAL);
}
inline CaffeLogger LOG_DFATAL(const std::string &severity)
{
  (void)severity;
  return LOG(FATAL);
}
inline CaffeLogger DLOG(const std::string &severity)
{
  return LOG(severity);
}
#endif

// Poor man's version...
inline CaffeLogger LOG_EVERY_N(const std::string &severity, const int &n)
{
  (void)n;
  return LOG(severity);
}

#endif
