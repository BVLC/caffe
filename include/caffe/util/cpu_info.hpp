#ifndef CAFFE_UTIL_CPU_INFO_HPP
#define CAFFE_UTIL_CPU_INFO_HPP

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

  Processor();
};

class Collection {
 public:
  static unsigned getTotalNumberOfSockets();
  static unsigned getTotalNumberOfCpuCores();
  static unsigned getNumberOfProcessors();
  static const Processor &getProcessor(unsigned processorId);

 private:
  unsigned totalNumberOfSockets;
  unsigned totalNumberOfCpuCores;
  std::vector<Processor> processors;
  Processor *currentProcessor;

  Collection();
  Collection(const Collection &collection);
  Collection &operator =(const Collection &collection);
  static Collection &getSingleInstance();

  void parseCpuFile(const char *fileName);
  void parseCpuFileContent(FILE *file);
  void parseCpuFileLine(const char *lineBuffer);
  void parseValue(const char *fieldName, const char *valueString);
  void appendNewProcessor();
  bool beginsWith(const char *lineBuffer, const char *text) const;
  void parseInteger(unsigned *value, const char *text) const;

  void collectBasicCpuInformation();
  void updateCpuInformation(const Processor &processor,
    unsigned numberOfUniquePhysicalId);
};

#ifdef _OPENMP

class OpenMpManager {
 public:
  static void setGpuEnabled();
  static void setGpuDisabled();

  static void bindCurrentThreadToPrimaryCore();
  static void bindOpenMpThreads();

  static void printVerboseInformation();

 private:
  bool isGpuEnabled;
  bool isAnyOpenMpEnvVarSpecified;
  cpu_set_t currentCpuSet;
  cpu_set_t currentCoreSet;

  OpenMpManager();
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
