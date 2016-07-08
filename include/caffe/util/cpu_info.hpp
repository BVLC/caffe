#ifndef CAFFE_UTIL_CPU_INFO_HPP
#define CAFFE_UTIL_CPU_INFO_HPP

#include <boost/thread/thread.hpp>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>


namespace caffe {
namespace cpu {

struct Processor {
  unsigned processor;
  unsigned physicalId;
  unsigned siblings;
  unsigned coreId;
  unsigned cpuCores;
  unsigned speedMHz;

  Processor();
};

class CpuInfoInterface {
 public:
  virtual ~CpuInfoInterface() {}
  virtual const char *getFirstLine() = 0;
  virtual const char *getNextLine() = 0;
};

class CpuInfo : public CpuInfoInterface {
 public:
  CpuInfo();
  explicit CpuInfo(const char *content);
  virtual ~CpuInfo();

  virtual const char *getFirstLine();
  virtual const char *getNextLine();

 private:
  const char *fileContentBegin;
  const char *fileContentEnd;
  const char *currentLine;

  void loadContentFromFile(const char *fileName);
  void loadContent(const char *content);
  void parseLines(char *content);
};

class CollectionInterface {
 public:
  virtual ~CollectionInterface() {}
  virtual unsigned getProcessorSpeedMHz() = 0;
  virtual unsigned getTotalNumberOfSockets() = 0;
  virtual unsigned getTotalNumberOfCpuCores() = 0;
  virtual unsigned getNumberOfProcessors() = 0;
  virtual const Processor &getProcessor(unsigned processorId) = 0;
};

class Collection : public CollectionInterface {
 public:
  explicit Collection(CpuInfoInterface *cpuInfo);

  virtual unsigned getProcessorSpeedMHz();
  virtual unsigned getTotalNumberOfSockets();
  virtual unsigned getTotalNumberOfCpuCores();
  virtual unsigned getNumberOfProcessors();
  virtual const Processor &getProcessor(unsigned processorId);

 private:
  CpuInfoInterface &cpuInfo;
  unsigned totalNumberOfSockets;
  unsigned totalNumberOfCpuCores;
  std::vector<Processor> processors;
  Processor *currentProcessor;

  Collection(const Collection &collection);
  Collection &operator =(const Collection &collection);

  void parseCpuInfo();
  void parseCpuInfoLine(const char *cpuInfoLine);
  void parseValue(const char *fieldName, const char *valueString);
  void appendNewProcessor();
  bool beginsWith(const char *lineBuffer, const char *text) const;
  unsigned parseInteger(const char *text) const;
  unsigned extractSpeedFromModelName(const char *text) const;

  void collectBasicCpuInformation();
  void updateCpuInformation(const Processor &processor,
    unsigned numberOfUniquePhysicalId);
};

#ifdef _OPENMP

class OpenMpManager {
 public:
  static void setGpuEnabled();
  static void setGpuDisabled();

  static void bindCurrentThreadToNonPrimaryCoreIfPossible();

  static void bindOpenMpThreads();
  static void printVerboseInformation();

  static bool isMajorThread(boost::thread::id currentThread);
  static unsigned getProcessorSpeedMHz();

 private:
  boost::thread::id mainThreadId;
  Collection &collection;

  bool isGpuEnabled;
  bool isAnyOpenMpEnvVarSpecified;
  cpu_set_t currentCpuSet;
  cpu_set_t currentCoreSet;

  explicit OpenMpManager(Collection *collection);
  OpenMpManager(const OpenMpManager &openMpManager);
  OpenMpManager &operator =(const OpenMpManager &openMpManager);
  static OpenMpManager &getInstance();

  void getOpenMpEnvVars();
  void getCurrentCpuSet();
  void getDefaultCpuSet(cpu_set_t *defaultCpuSet);
  void getCurrentCoreSet();

  void selectAllCoreCpus(cpu_set_t *set, unsigned physicalCoreId);
  unsigned getPhysicalCoreId(unsigned logicalCoreId);

  bool isThreadsBindAllowed();
  void setOpenMpThreadNumberLimit();
  void bindCurrentThreadToLogicalCoreCpu(unsigned logicalCoreId);
  void bindCurrentThreadToLogicalCoreCpus(unsigned logicalCoreId);
};

#endif  // _OPENMP

}  // namespace cpu

}  // namespace caffe

#endif  // CAFFE_UTIL_CPU_INFO_HPP
