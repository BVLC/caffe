#ifndef CAFFE_UTIL_CPU_INFO_HPP
#define CAFFE_UTIL_CPU_INFO_HPP

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

  Collection(const Collection &collection);
  Collection &operator =(const Collection &collection);

  static Collection &getSingleInstance();

  Collection();

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
  static void applyConfiguration();

 private:
  bool isGpuEnabled;
  bool areEnvVarsSpecified;

  OpenMpManager(const OpenMpManager &collection);
  OpenMpManager &operator =(const OpenMpManager &collection);

  static OpenMpManager &getInstance();

  OpenMpManager();
  void testForEnvVariablePresence(const char *envVariableName);
  void bindOpenMpCores();
  unsigned getRecommendedNumberOfOpenMpThreads();
  void setCpuAffinityThreadLimit(unsigned usedProcessorsLimit);
  void printVerboseInformation(unsigned recommendedNumberOfOpenMpThreads);
};

#endif  // _OPENMP

}  // namespace cpu

}  // namespace caffe

#endif  // CAFFE_UTIL_CPU_INFO_HPP
