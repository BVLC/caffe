/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of Intel Corporation nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <glog/logging.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "caffe/util/cpu_info.hpp"

#if defined(_MSC_EXTENSIONS)
#include <Powrprof.h>
#pragma comment(lib, "Powrprof.lib")
#endif

namespace caffe {
namespace cpu {

Processor::Processor() {
  processor = 0;
  physicalId = 0;
  siblings = 0;
  coreId = 0;
  cpuCores = 0;
  speedMHz = 0;
  mask = 0;
}

CpuInfo::CpuInfo() {
#if !defined(_MSC_EXTENSIONS)
  loadContentFromFile("/proc/cpuinfo");
#endif
}

CpuInfo::CpuInfo(const char *content) {
  std::string str_content(content);
  loadContent(str_content);
}

void CpuInfo::loadContentFromFile(const char *fileName) {
  std::ifstream file(fileName);
  std::string content(
    (std::istreambuf_iterator<char>(file)),
    (std::istreambuf_iterator<char>()));

  loadContent(content);
}

void CpuInfo::loadContent(std::string &content) {
  size_t contentLength = content.length();
  char *contentCopy = new char[contentLength + 1];
  snprintf(contentCopy, contentLength + 1, "%s", content.c_str());

  parseLines(contentCopy);

  fileContentBegin = contentCopy;
  fileContentEnd = &contentCopy[contentLength];
  currentLine = NULL;
}

CpuInfo::~CpuInfo() {
#if !defined(_MSC_EXTENSIONS)
  delete[] fileContentBegin;
#endif
}

void CpuInfo::parseLines(char *content) {
  for (; *content; content++) {
    if (*content == '\n') {
      *content = '\0';
    }
  }
}

const char *CpuInfo::getFirstLine() {
  currentLine = fileContentBegin < fileContentEnd ? fileContentBegin : NULL;
  return getNextLine();
}

const char *CpuInfo::getNextLine() {
  if (!currentLine) {
    return NULL;
  }

  const char *savedCurrentLine = currentLine;
  while (*(currentLine++)) {
  }

  if (currentLine >= fileContentEnd) {
    currentLine = NULL;
  }

  return savedCurrentLine;
}

#if defined(_MSC_EXTENSIONS)
typedef BOOL(WINAPI *LPFN_GLPI)(
  LOGICAL_PROCESSOR_RELATIONSHIP,
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
  PDWORD);

typedef struct _PROCESSOR_POWER_INFORMATION {
  ULONG Number;
  ULONG MaxMhz;
  ULONG CurrentMhz;
  ULONG MhzLimit;
  ULONG MaxIdleState;
  ULONG CurrentIdleState;
} PROCESSOR_POWER_INFORMATION, *PPROCESSOR_POWER_INFORMATION;

// Helper function to count set bits in the processor mask.
DWORD CountSetBits(ULONG_PTR bitMask)
{
  DWORD LSHIFT = sizeof(ULONG_PTR) * 8 - 1;
  DWORD bitSetCount = 0;
  ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;
  DWORD i;

  for (i = 0; i <= LSHIFT; ++i) {
    bitSetCount += ((bitMask & bitTest) ? 1 : 0);
    bitTest /= 2;
  }

  return bitSetCount;
}

void Collection::parseCpuInfo() {
  LPFN_GLPI glpi;
  BOOL done = FALSE;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = NULL;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX ptr = NULL;
  DWORD returnLength = 0;
  DWORD logicalProcessorCount = 0;
  DWORD numaNodeCount = 0;
  DWORD processorCoreCount = 0;
  DWORD processorPackageCount = 0;
  DWORD byteOffset = 0;

  glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformationEx");
  if (NULL == glpi) {
    _tprintf(TEXT("\nGetLogicalProcessorInformation is not supported.\n"));
    return;
  }

  while (!done) {
    DWORD rc = glpi(RelationAll, buffer, &returnLength);
    if (FALSE == rc) {
      if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        if (buffer)
          free(buffer);
        buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(returnLength);
        if (NULL == buffer) {
          _tprintf(TEXT("\nError: Allocation failure\n"));
          return;
        }
      }
      else {
        _tprintf(TEXT("\nError %d\n"), GetLastError());
        return;
      }
    }
    else {
      done = TRUE;
    }
  }

  ptr = buffer;
  while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= returnLength) {
    switch (ptr->Relationship) {
    case RelationNumaNode:
      // Non-NUMA systems report a single record of this type.
      numaNodeCount++;
      break;
    case RelationProcessorCore:
      processorCoreCount++;
      // A hyperthreaded core supplies more than one logical processor.
      logicalProcessorCount += CountSetBits(ptr->Processor.GroupMask[0].Mask);

      for (unsigned size = 0; size < CountSetBits(ptr->Processor.GroupMask[0].Mask); size++) {
        processors.push_back(Processor());
        Processor *processor = &processors.back();
        processor->processor = logicalProcessorCount - CountSetBits(ptr->Processor.GroupMask[0].Mask) + size;
        processor->physicalId = ptr->Processor.GroupMask[0].Group;
        processor->cpuCores = processorCoreCount - 1;
        processor->mask = ptr->Processor.GroupMask[0].Mask;
      }
      break;
    case RelationProcessorPackage:
      // Logical processors share a physical package.
      processorPackageCount++;
      break;
    default:
      break;
    }
    byteOffset += ptr->Size;
    ptr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((LPBYTE)ptr + ptr->Size);
  }
  free(buffer);

  if (getNumberOfProcessors() == 0) {
    _tprintf(TEXT("\nError: Get Processor Info fails\n"));
    return;
  }
  DWORD dwSize = sizeof(PROCESSOR_POWER_INFORMATION) * getNumberOfProcessors();
  PPROCESSOR_POWER_INFORMATION powerInfo = NULL;

  powerInfo = (PPROCESSOR_POWER_INFORMATION)malloc(dwSize);
  if (NULL == powerInfo) {
    _tprintf(TEXT("\nError: Allocation PowerInfo failure\n"));
    return;
  }

  while (!CallNtPowerInformation(ProcessorInformation, NULL, 0, powerInfo, dwSize)) {
    processors[0].speedMHz = powerInfo->MaxMhz;
    free(powerInfo);
    return;
  }
  return;
}
#else
void Collection::parseCpuInfo() {
  const char *cpuInfoLine = cpuInfo.getFirstLine();
  for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {
    parseCpuInfoLine(cpuInfoLine);
  }
}

void Collection::parseCpuInfoLine(const char *cpuInfoLine) {
  int delimiterPosition = strcspn(cpuInfoLine, ":");

  if (cpuInfoLine[delimiterPosition] == '\0') {
    currentProcessor = NULL;
  }
  else {
    parseValue(cpuInfoLine, &cpuInfoLine[delimiterPosition + 2]);
  }
}

void Collection::parseValue(const char *fieldName, const char *valueString) {
  if (!currentProcessor) {
    appendNewProcessor();
  }

  if (beginsWith(fieldName, "processor")) {
    currentProcessor->processor = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "physical id")) {
    currentProcessor->physicalId = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "siblings")) {
    currentProcessor->siblings = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "core id")) {
    currentProcessor->coreId = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "cpu cores")) {
    currentProcessor->cpuCores = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "model name")) {
    currentProcessor->speedMHz = extractSpeedFromModelName(valueString);
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

unsigned Collection::parseInteger(const char *text) const {
  return atol(text);
}

/* Function extracts CPU speed from model name. If unit is not set it is
   assumed that values below 100 are specified in GHz, otherwise MHz */
unsigned Collection::extractSpeedFromModelName(const char *text) const {
  text = strstr(text, "@");
  if (!text) {
    return 0;
  }

  char *unit;
  double speed = strtod(&text[1], &unit);

  while (isspace(*unit)) {
    unit++;
  }

  bool isMHz = !strncmp(unit, "MHz", 3);
  bool isGHz = !strncmp(unit, "GHz", 3);
  bool isGHzPossible = (speed < 100);

  if (isGHz || (isGHzPossible && !isMHz)) {
    return 1000 * speed + 0.5;
  }
  else {
    return speed + 0.5;
  }
}
#endif
Collection::Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {
  totalNumberOfSockets = 0;
  totalNumberOfCpuCores = 0;
  currentProcessor = NULL;

  processors.reserve(96);

  parseCpuInfo();
  collectBasicCpuInformation();
}

unsigned Collection::getNumberOfProcessors() {
  return processors.size();
}

const Processor &Collection::getProcessor(unsigned processorId) {
  return processors[processorId];
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
#if !defined(_MSC_EXTENSIONS)
  if (totalNumberOfSockets == numberOfUniquePhysicalId) {
    return;
  }
#endif
  totalNumberOfSockets = numberOfUniquePhysicalId;
#if !defined(_MSC_EXTENSIONS)
  totalNumberOfCpuCores += processor.cpuCores;
#else
  totalNumberOfCpuCores = processor.cpuCores + 1;
#endif
}

unsigned Collection::getTotalNumberOfSockets() {
  return totalNumberOfSockets;
}

unsigned Collection::getTotalNumberOfCpuCores() {
  return totalNumberOfCpuCores;
}

unsigned Collection::getProcessorSpeedMHz() {
  return processors.size() ? processors[0].speedMHz : 0;
}
#ifdef _OPENMP

/* The OpenMpManager class is responsible for determining a set of all of
   available CPU cores and delegating each core to perform other tasks. The
   first of available cores is delegated for background threads, while other
   remaining cores are dedicated for OpenMP threads. Each OpenMP thread owns
   one core for exclusive use. The number of OpenMP threads is then limited
   to the number of available cores minus one. The amount of CPU cores may
   be limited by system eg. when numactl was used. */

#include <omp.h>

static const char *openMpEnvVars[] = {
  "OMP_CANCELLATION", "OMP_DISPLAY_ENV", "OMP_DEFAULT_DEVICE", "OMP_DYNAMIC",
  "OMP_MAX_ACTIVE_LEVELS", "OMP_MAX_TASK_PRIORITY", "OMP_NESTED",
  "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_STACKSIZE",
  "OMP_SCHEDULE", "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_CPU_AFFINITY",
  "GOMP_DEBUG", "GOMP_STACKSIZE", "GOMP_SPINCOUNT", "GOMP_RTEMS_THREAD_POOLS",
  "KMP_AFFINITY", "KMP_NUM_THREADS", "MIC_KMP_AFFINITY",
  "MIC_OMP_NUM_THREADS", "MIC_OMP_PROC_BIND", "PHI_KMP_AFFINITY",
  "PHI_OMP_NUM_THREADS", "PHI_KMP_PLACE_THREADS", "MKL_NUM_THREADS",
  "MKL_DYNAMIC", "MKL_DOMAIN_NUM_THREADS"
};

static const unsigned numberOfOpenMpEnvVars =
  sizeof(openMpEnvVars) / sizeof(openMpEnvVars[0]);

OpenMpManager::OpenMpManager(Collection *collection) :
  mainThreadId(boost::this_thread::get_id()),
  collection(*collection) {
  getOpenMpEnvVars();
  getCurrentCpuSet();
  getCurrentCoreSet();
}

OpenMpManager &OpenMpManager::getInstance() {
  static CpuInfo cpuInfo;
  static Collection collection(&cpuInfo);
  static OpenMpManager openMpManager(&collection);
  return openMpManager;
}

void OpenMpManager::setGpuEnabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = true;
}

void OpenMpManager::setGpuDisabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = false;
}

bool OpenMpManager::isMajorThread(boost::thread::id currentThread) {
  OpenMpManager &openMpManager = getInstance();
  return (boost::this_thread::get_id() == openMpManager.mainThreadId);
}

// Ideally bind given thread to secondary logical core, if
// only one thread exists then bind to primary one
void OpenMpManager::bindCurrentThreadToNonPrimaryCoreIfPossible() {
  OpenMpManager &openMpManager = getInstance();
  if (openMpManager.isThreadsBindAllowed()) {
#if defined(_MSC_EXTENSIONS)
    int totalNumberOfAvailableCores = 0;
    totalNumberOfAvailableCores += CountSetBits(openMpManager.currentCoreSet.Mask[0]);
    totalNumberOfAvailableCores += CountSetBits(openMpManager.currentCoreSet.Mask[1]);
    totalNumberOfAvailableCores += CountSetBits(openMpManager.currentCoreSet.Mask[2]);
    totalNumberOfAvailableCores += CountSetBits(openMpManager.currentCoreSet.Mask[3]);
#else
    int totalNumberOfAvailableCores = CPU_COUNT(&openMpManager.currentCoreSet);
#endif
    int logicalCoreToBindTo = totalNumberOfAvailableCores > 1 ? 1 : 0;
    openMpManager.bindCurrentThreadToLogicalCoreCpus(logicalCoreToBindTo);
  }
}

void OpenMpManager::bindOpenMpThreads() {
  OpenMpManager &openMpManager = getInstance();

  if (!openMpManager.isThreadsBindAllowed())
    return;

  openMpManager.setOpenMpThreadNumberLimit();
#pragma omp parallel
  {
    unsigned logicalCoreId = omp_get_thread_num();
    openMpManager.bindCurrentThreadToLogicalCoreCpu(logicalCoreId);
  }
}

void OpenMpManager::getOpenMpEnvVars() {
  isAnyOpenMpEnvVarSpecified = false;
  for (unsigned i = 0; i < numberOfOpenMpEnvVars; i++) {
    if (getenv(openMpEnvVars[i])) {
      isAnyOpenMpEnvVarSpecified = true;
    }
  }
}

void OpenMpManager::getCurrentCpuSet() {
#if defined(_MSC_EXTENSIONS)
  // unlike linux, the thread group affinity of windows is by default only allowed on the
  // physical cores of one socket.
  // To get the same behavior of linux, we have to avoid using GetThreadGroupAffinity() here
  getDefaultCpuSet(&currentCpuSet);
#else
  if (sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet)) {
    getDefaultCpuSet(&currentCpuSet);
  }
#endif
}

void OpenMpManager::getDefaultCpuSet(cpu_set_t *defaultCpuSet) {
#if defined(_MSC_EXTENSIONS)
  ZeroMemory(defaultCpuSet, sizeof(cpu_set_t));
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  for (unsigned processorId = 0; processorId < numberOfProcessors; processorId++) {
    defaultCpuSet->Mask[collection.getProcessor(processorId).physicalId] |= collection.getProcessor(processorId).mask;
  }
#else
  CPU_ZERO(defaultCpuSet);
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    CPU_SET(processorId, defaultCpuSet);
  }
#endif
}

/* Function getCurrentCoreSet() fills currentCoreSet variable with a set of
   available CPUs, where only one CPU per core is chosen. When multiple CPUs
   of single core are used, function is selecting only first one of all
   available. */

void OpenMpManager::getCurrentCoreSet() {
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

#if defined(_MSC_EXTENSIONS)
  CopyMemory(&currentCoreSet, &currentCpuSet, sizeof(cpu_set_t));

  int percore = numberOfProcessors / totalNumberOfCpuCores;
  for (int idx = 0; idx < sizeof(cpu_set_t) / sizeof(size_t); idx++) {
    size_t *mask = &currentCoreSet.Mask[idx];
    size_t bit_mask = (pow(2, percore) - 1);
    size_t bit_value = 1;

    for (int bit_idx = 0; bit_idx < sizeof(size_t) * 8 / percore; bit_idx++) {
      if ((*mask) & bit_mask) {
        *mask &= ~bit_mask;
        *mask |= bit_value;
      }
      bit_mask <<= percore;
      bit_value <<= percore;
    }
  }
#else
  cpu_set_t usedCoreSet;
  CPU_ZERO(&usedCoreSet);
  CPU_ZERO(&currentCoreSet);

  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, &currentCpuSet)) {
      unsigned coreId = processorId % totalNumberOfCpuCores;
      if (!CPU_ISSET(coreId, &usedCoreSet)) {
        CPU_SET(coreId, &usedCoreSet);
        CPU_SET(processorId, &currentCoreSet);
      }
    }
  }
#endif
}

void OpenMpManager::selectAllCoreCpus(cpu_set_t *set, unsigned physicalCoreId) {
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

#if !defined(_MSC_EXTENSIONS)
  int processorId = physicalCoreId % totalNumberOfCpuCores;
  while (processorId < numberOfProcessors) {
    if (CPU_ISSET(processorId, &currentCpuSet)) {
      CPU_SET(processorId, set);
    }

    processorId += totalNumberOfCpuCores;
  }
#endif
}

unsigned OpenMpManager::getPhysicalCoreId(unsigned logicalCoreId) {
#if defined(_MSC_EXTENSIONS)
  size_t mask = 0;
  int processorId = 0;
  int persocket = collection.getNumberOfProcessors() / collection.getTotalNumberOfSockets();
  do {
    for (int idx = 0; idx < sizeof(cpu_set_t) / sizeof(size_t); idx++) {
      mask = currentCoreSet.Mask[idx];
      for (int bit = 0; bit < persocket; bit++) {
        if ((mask >> bit) & 0x1) {
          if (!logicalCoreId--) {
            return processorId;
          }
        }
        processorId++;
      }
    }
  } while (logicalCoreId);
#else
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, &currentCoreSet)) {
      if (!logicalCoreId--) {
        return processorId;
      }
    }
  }
#endif
  LOG(FATAL) << "This should never happen!";
  return 0;
}

bool OpenMpManager::isThreadsBindAllowed() {
  return !isAnyOpenMpEnvVarSpecified && !isGpuEnabled;
}

// Limit of threads to number of logical cores available
void OpenMpManager::setOpenMpThreadNumberLimit() {
#if defined(_MSC_EXTENSIONS)
  unsigned short num = 0;
  for (int idx = 0; idx < sizeof(cpu_set_t) / sizeof(size_t); idx++) {
    num += CountSetBits(currentCoreSet.Mask[idx]);
  }
  omp_set_num_threads(collection.getTotalNumberOfCpuCores());
#else
  omp_set_num_threads(CPU_COUNT(&currentCoreSet));
#endif
}

void OpenMpManager::bindCurrentThreadToLogicalCoreCpu(unsigned logicalCoreId) {
#if defined(_MSC_EXTENSIONS)
  HANDLE handle = GetCurrentThread();
  GROUP_AFFINITY groupAffinity;
  ZeroMemory(&groupAffinity, sizeof(GROUP_AFFINITY));
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);
  int persocket = collection.getNumberOfProcessors() / collection.getTotalNumberOfSockets();
  groupAffinity.Mask = 1ull << (physicalCoreId % persocket);
  groupAffinity.Group = physicalCoreId / persocket;
#pragma omp critical
  SetThreadGroupAffinity(handle, &groupAffinity, NULL);
#else
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(physicalCoreId, &set);
  sched_setaffinity(0, sizeof(set), &set);
#endif
}

void OpenMpManager::bindCurrentThreadToLogicalCoreCpus(unsigned logicalCoreId) {
#if defined(_MSC_EXTENSIONS)
  HANDLE handle = GetCurrentThread();
  GROUP_AFFINITY groupAffinity;
  ZeroMemory(&groupAffinity, sizeof(GROUP_AFFINITY));
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);
  int persocket = collection.getNumberOfProcessors() / collection.getTotalNumberOfSockets();
  int percore = collection.getNumberOfProcessors() / collection.getTotalNumberOfCpuCores();
  groupAffinity.Mask = (size_t)(pow(2, percore) - 1) << (physicalCoreId % persocket);
  groupAffinity.Group = physicalCoreId / persocket;
  SetThreadGroupAffinity(handle, &groupAffinity, NULL);
#else
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);
  cpu_set_t set;
  CPU_ZERO(&set);
  selectAllCoreCpus(&set, physicalCoreId);
  sched_setaffinity(0, sizeof(set), &set);
#endif  
}

void OpenMpManager::printVerboseInformation() {
  OpenMpManager &openMpManager = getInstance();

  LOG(INFO) << "Processor speed [MHz]: "
    << openMpManager.collection.getProcessorSpeedMHz();

  LOG(INFO) << "Total number of sockets: "
    << openMpManager.collection.getTotalNumberOfSockets();

  LOG(INFO) << "Total number of CPU cores: "
    << openMpManager.collection.getTotalNumberOfCpuCores();

  LOG(INFO) << "Total number of processors: "
    << openMpManager.collection.getNumberOfProcessors();

  LOG(INFO) << "GPU is used: "
    << (openMpManager.isGpuEnabled ? "yes" : "no");

  LOG(INFO) << "OpenMP environmental variables are specified: "
    << (openMpManager.isAnyOpenMpEnvVarSpecified ? "yes" : "no");

  LOG(INFO) << "OpenMP thread bind allowed: "
    << (openMpManager.isThreadsBindAllowed() ? "yes" : "no");

  LOG(INFO) << "Number of OpenMP threads: "
    << omp_get_max_threads();
}

unsigned OpenMpManager::getProcessorSpeedMHz() {
  OpenMpManager &openMpManager = getInstance();
  return openMpManager.collection.getProcessorSpeedMHz();
}

unsigned OpenMpManager::getNumaNode() {
#if defined(_MSC_EXTENSIONS)
  PROCESSOR_NUMBER num;
  USHORT node;
  BOOL   success;
  GetCurrentProcessorNumberEx(&num);
  success = GetNumaProcessorNodeEx(&num, &node);
  assert(success != 0);
  return node;
#else
  OpenMpManager &openMpManager = getInstance();
  int cpu = sched_getcpu();
  assert(cpu >= 0);
  return openMpManager.collection.getProcessor(cpu).physicalId;
#endif
}
#endif  // _OPENMP

}  // namespace cpu
}  // namespace caffe
