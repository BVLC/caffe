#include <glog/logging.h>
#include <set>
#include <vector>
#include "caffe/util/cpu_info.hpp"

namespace caffe {
namespace cpu {

Processor::Processor() {
  processor = 0;
  physicalId = 0;
  siblings = 0;
  coreId = 0;
  cpuCores = 0;
}

unsigned Collection::getTotalNumberOfSockets() {
  Collection &collection = getSingleInstance();
  return collection.totalNumberOfSockets;
}

unsigned Collection::getTotalNumberOfCpuCores() {
  Collection &collection = getSingleInstance();
  return collection.totalNumberOfCpuCores;
}

unsigned Collection::getNumberOfProcessors() {
  Collection &collection = getSingleInstance();
  return collection.processors.size();
}

const Processor &Collection::getProcessor(unsigned processorId) {
  Collection &collection = getSingleInstance();
  return collection.processors[processorId];
}

Collection &Collection::getSingleInstance() {
  static Collection collection;
  return collection;
}

Collection::Collection() {
  totalNumberOfSockets = 0;
  totalNumberOfCpuCores = 0;
  currentProcessor = NULL;

  processors.reserve(96);

  parseCpuFile("/proc/cpuinfo");
  collectBasicCpuInformation();
}

void Collection::parseCpuFile(const char *fileName) {
  FILE *file = fopen(fileName, "rb");
  if (!file) {
    return;
  }

  parseCpuFileContent(file);

  fclose(file);
}

void Collection::parseCpuFileContent(FILE *file) {
  while (!feof(file)) {
    char lineBuffer[1024];
    fgets(lineBuffer, sizeof(lineBuffer), file);
    parseCpuFileLine(lineBuffer);
  }
}

void Collection::parseCpuFileLine(const char *lineBuffer) {
  int delimiterPosition = strcspn(lineBuffer, ":");

  if (lineBuffer[delimiterPosition] == '\0') {
    currentProcessor = NULL;
  } else {
    parseValue(lineBuffer, &lineBuffer[delimiterPosition + 2]);
  }
}

void Collection::parseValue(const char *fieldName, const char *valueString) {
  if (!currentProcessor) {
    appendNewProcessor();
  }

  if (beginsWith(fieldName, "processor")) {
    return parseInteger(&currentProcessor->processor, valueString);
  }

  if (beginsWith(fieldName, "physical id")) {
    return parseInteger(&currentProcessor->physicalId, valueString);
  }

  if (beginsWith(fieldName, "siblings")) {
    return parseInteger(&currentProcessor->siblings, valueString);
  }

  if (beginsWith(fieldName, "core id")) {
    return parseInteger(&currentProcessor->coreId, valueString);
  }

  if (beginsWith(fieldName, "cpu cores")) {
    return parseInteger(&currentProcessor->cpuCores, valueString);
  }
}

void Collection::appendNewProcessor() {
  processors.push_back(Processor());
  currentProcessor = &processors.back();
}

bool Collection::beginsWith(const char *lineBuffer, const char *text) const {
  while (*text) {
    if (*(lineBuffer++) != *(text++)) {
      return false;
    }
  }

  return true;
}

void Collection::parseInteger(unsigned *value, const char *text) const {
  *value = atol(text);
}

void Collection::collectBasicCpuInformation() {
  std::set<unsigned> uniquePhysicalId;
  std::vector<Processor>::iterator processor = processors.begin();
  for (; processor != processors.end(); processor++) {
    uniquePhysicalId.insert(processor->physicalId);
    updateCpuInformation(*processor, uniquePhysicalId.size());
  }
}

void Collection::updateCpuInformation(const Processor &processor,
    unsigned numberOfUniquePhysicalId) {
  if (totalNumberOfSockets == numberOfUniquePhysicalId) {
    return;
  }

  totalNumberOfSockets = numberOfUniquePhysicalId;
  totalNumberOfCpuCores += processor.cpuCores;
}

#ifdef _OPENMP

#include <omp.h>
#include <sched.h>

void OpenMpManager::setGpuEnabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = true;
}

void OpenMpManager::setGpuDisabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = false;
}

void OpenMpManager::applyConfiguration() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.bindOpenMpCores();
}

OpenMpManager &OpenMpManager::getInstance() {
  static OpenMpManager openMpManager;
  return openMpManager;
}

OpenMpManager::OpenMpManager() {
  isGpuEnabled = false;
  areEnvVarsSpecified = false;

  testForEnvVariablePresence("OMP_CANCELLATION");
  testForEnvVariablePresence("OMP_DISPLAY_ENV");
  testForEnvVariablePresence("OMP_DEFAULT_DEVICE");
  testForEnvVariablePresence("OMP_DYNAMIC");
  testForEnvVariablePresence("OMP_MAX_ACTIVE_LEVELS");
  testForEnvVariablePresence("OMP_MAX_TASK_PRIORITY");
  testForEnvVariablePresence("OMP_NESTED");
  testForEnvVariablePresence("OMP_NUM_THREADS");
  testForEnvVariablePresence("OMP_PROC_BIND");
  testForEnvVariablePresence("OMP_PLACES");
  testForEnvVariablePresence("OMP_STACKSIZE");
  testForEnvVariablePresence("OMP_SCHEDULE");
  testForEnvVariablePresence("OMP_THREAD_LIMIT");
  testForEnvVariablePresence("OMP_WAIT_POLICY");
  testForEnvVariablePresence("GOMP_CPU_AFFINITY");
  testForEnvVariablePresence("GOMP_DEBUG");
  testForEnvVariablePresence("GOMP_STACKSIZE");
  testForEnvVariablePresence("GOMP_SPINCOUNT");
  testForEnvVariablePresence("GOMP_RTEMS_THREAD_POOLS");

  testForEnvVariablePresence("KMP_AFFINITY");
  testForEnvVariablePresence("KMP_NUM_THREADS");
  testForEnvVariablePresence("MIC_KMP_AFFINITY");
  testForEnvVariablePresence("MIC_OMP_NUM_THREADS");
  testForEnvVariablePresence("PHI_KMP_AFFINITY");
  testForEnvVariablePresence("PHI_OMP_NUM_THREADS");
  testForEnvVariablePresence("MIC_OMP_PROC_BIND");
  testForEnvVariablePresence("PHI_KMP_PLACE_THREADS");
  testForEnvVariablePresence("MKL_NUM_THREADS");
  testForEnvVariablePresence("MKL_DYNAMIC");
  testForEnvVariablePresence("MKL_DOMAIN_NUM_THREADS");
}

void OpenMpManager::testForEnvVariablePresence(const char *envVariableName) {
  if (getenv(envVariableName)) {
    areEnvVarsSpecified = true;
  }
}

void OpenMpManager::bindOpenMpCores() {
  unsigned recommendedNumberOfOpenMpThreads =
    getRecommendedNumberOfOpenMpThreads();
  unsigned usedProcessorsLimit = recommendedNumberOfOpenMpThreads;

  if (!areEnvVarsSpecified) {
    usedProcessorsLimit = setCpuAffinityThreadLimit(usedProcessorsLimit);
  }

  printVerboseInformation(recommendedNumberOfOpenMpThreads,
    usedProcessorsLimit);
}

unsigned OpenMpManager::getRecommendedNumberOfOpenMpThreads() {
  unsigned totalNumberOfCpuCores = Collection::getTotalNumberOfCpuCores();
  unsigned maximalNumberOfOpenMpThreads = omp_get_max_threads();

  if (isGpuEnabled) {
    return 1;
  }

  if (areEnvVarsSpecified) {
    return omp_get_num_threads();
  }

  return totalNumberOfCpuCores < maximalNumberOfOpenMpThreads ?
    totalNumberOfCpuCores : maximalNumberOfOpenMpThreads;
}

void OpenMpManager::getCoreMask(cpu_set_t *core_set, cpu_set_t *current_set) {
  unsigned numberOfProcessors = Collection::getNumberOfProcessors();
  unsigned totalNumberOfCpuCores = Collection::getTotalNumberOfCpuCores();

  cpu_set_t used_cores_set;
  CPU_ZERO(&used_cores_set);

  CPU_ZERO(core_set);
  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, current_set)) {
      unsigned coreId = processorId % totalNumberOfCpuCores;
      if (!CPU_ISSET(coreId, &used_cores_set)) {
        CPU_SET(coreId, &used_cores_set);
        CPU_SET(processorId, core_set);
      }
    }
  }
}

unsigned OpenMpManager::getPhysicalCoreId(cpu_set_t *core_set,
    unsigned logicalId) {
  unsigned numberOfProcessors = Collection::getNumberOfProcessors();
  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, core_set)) {
      if (!logicalId--) {
        return processorId;
      }
    }
  }

  LOG(FATAL) << "This should never occur!";
  return 0;
}

unsigned OpenMpManager::setCpuAffinityThreadLimit(
    unsigned usedProcessorsLimit) {
  cpu_set_t current_set;
  if (sched_getaffinity(0, sizeof(current_set), &current_set))
    return 0;

  cpu_set_t core_set;
  getCoreMask(&core_set, &current_set);

  unsigned numberOfAvailableCores = CPU_COUNT(&core_set);
  if (usedProcessorsLimit > numberOfAvailableCores) {
    usedProcessorsLimit = numberOfAvailableCores;
  }

  omp_set_num_threads(usedProcessorsLimit);

  #pragma omp parallel
  {
    unsigned logicalCoreId = omp_get_thread_num();
    unsigned physicalCoreId = getPhysicalCoreId(&core_set, logicalCoreId);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(physicalCoreId, &set);
    sched_setaffinity(0, sizeof(set), &set);
  }

  return usedProcessorsLimit;
}

void OpenMpManager::printVerboseInformation(
    unsigned recommendedNumberOfOpenMpThreads,
    unsigned usedProcessorsLimit) {
  LOG(INFO) << "Total number of sockets: "
    << Collection::getTotalNumberOfSockets();

  LOG(INFO) << "Total number of CPU cores: "
    << Collection::getTotalNumberOfCpuCores();

  LOG(INFO) << "Total number of processors: "
    << Collection::getNumberOfProcessors();

  LOG(INFO) << "GPU status: "
    << (isGpuEnabled ? "used" : "not used");

  LOG(INFO) << "OpenMP environmental variables: "
    << (areEnvVarsSpecified ? "specified" : "not specified");

  LOG(INFO) << "OpenMP thread bind status: "
    << ((recommendedNumberOfOpenMpThreads > 1) ? "enabled" : "disabled");

  LOG(INFO) << "Number of available OpenMP threads: "
    << omp_get_thread_limit();

  LOG(INFO) << "Recommended number of OpenMP threads: "
    << recommendedNumberOfOpenMpThreads;

  LOG(INFO) << "Number of OpenMP threads: "
    << usedProcessorsLimit;
}

#endif  // _OPENMP

}  // namespace cpu
}  // namespace caffe
