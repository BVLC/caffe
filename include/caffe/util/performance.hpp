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
#ifndef PerformanceH
#define PerformanceH

#ifdef PERFORMANCE_MONITORING

#define PERFORMANCE_EVENT_ID_UNSET (-1)

#define PERFORMANCE_EVENT_ID_DECL(id_name) \
  int id_name

#define PERFORMANCE_EVENT_ID_RESET(id_name) \
  id_name = PERFORMANCE_EVENT_ID_UNSET

#define PERFORMANCE_EVENT_ID_INIT(id_name, event_name) \
  if ((id_name) == PERFORMANCE_EVENT_ID_UNSET)         \
    id_name = performance::monitor.GetEventIdByName(event_name)

#define PERFORMANCE_MEASUREMENT_BEGIN()            \
  performance::Measurement m_MACRO;                \
  m_MACRO.Start();

#define PERFORMANCE_MEASUREMENT_END(name)                                \
  m_MACRO.Stop();                                                        \
  int id_MACRO = performance::monitor.GetEventIdByName(name);            \
  performance::monitor.UpdateEventById(id_MACRO, m_MACRO);

#define PERFORMANCE_MEASUREMENT_END_STATIC(name)                     \
  m_MACRO.Stop();                                                    \
  static int id_MACRO = performance::monitor.GetEventIdByName(name); \
  performance::monitor.UpdateEventById(id_MACRO, m_MACRO);

#define PERFORMANCE_MEASUREMENT_END_ID(id_name)           \
  m_MACRO.Stop();                                         \
  performance::monitor.UpdateEventById(id_name, m_MACRO);

#define PERFORMANCE_CREATE_MONITOR() \
  namespace performance {            \
  Monitor monitor; };

#define PERFORMANCE_INIT_MONITOR()           \
  performance::monitor.EnableMeasurements(); \
  performance::monitor.MarkAsInitialized();

#define PERFORMANCE_START_RESETTING_MONITOR() \
  performance::monitor.StartResetting();

#define PERFORMANCE_STOP_RESETTING_MONITOR() \
  performance::monitor.StopResetting();

#define PERFORMANCE_MEASUREMENT_END_MKL(prefix)       \
  do {                                                \
    static char name[256];                            \
    snprintf(name, sizeof(name), "%s_mkl_%s", prefix, \
      this->layer_param_.name().c_str());             \
    PERFORMANCE_MEASUREMENT_END(name);                \
  } while(0)

#define PERFORMANCE_MEASUREMENT_END_MKL_DETAILED(prefix, suffix) \
  do {                                                           \
    static char name[256];                                       \
    snprintf(name, sizeof(name), "%s_mkl_%s%s", prefix,          \
      this->layer_param_.name().c_str(), suffix);                \
    PERFORMANCE_MEASUREMENT_END(name);                           \
  } while(0)

#define PERFORMANCE_MKL_NAME_DETAILED(prefix, suffix)        \
  (std::string(prefix) + "_mkl_" + this->layer_param_.name() \
    + std::string(suffix)).c_str()

#define PERFORMANCE_MKL_NAME(prefix) \
  (std::string(prefix) + "_mkl_" + this->layer_param_.name()).c_str()

#define PERFORMANCE_MKLDNN_NAME_DETAILED(prefix, suffix)        \
  (std::string(prefix) + "_mkldnn_" + this->layer_param_.name() \
    + std::string(suffix)).c_str()

#define PERFORMANCE_MKLDNN_NAME(prefix) \
  (std::string(prefix) + "_mkldnn_" + this->layer_param_.name()).c_str()

#else
#define PERFORMANCE_EVENT_ID_DECL(id_name)
#define PERFORMANCE_EVENT_ID_RESET(id_name)
#define PERFORMANCE_EVENT_ID_INIT(id_name, event_name)
#define PERFORMANCE_MEASUREMENT_BEGIN()
#define PERFORMANCE_MEASUREMENT_END(name)
#define PERFORMANCE_MEASUREMENT_END_STATIC(name)
#define PERFORMANCE_MEASUREMENT_END_ID(id_name)
#define PERFORMANCE_CREATE_MONITOR()
#define PERFORMANCE_INIT_MONITOR()
#define PERFORMANCE_START_RESETTING_MONITOR()
#define PERFORMANCE_STOP_RESETTING_MONITOR()
#define PERFORMANCE_MEASUREMENT_END_MKL(prefix)
#define PERFORMANCE_MEASUREMENT_END_MKL_DETAILED(prefix, suffix)
#define PERFORMANCE_MKL_NAME_DETAILED(prefix, suffix)
#define PERFORMANCE_MKL_NAME(prefix)
#define PERFORMANCE_MKLDNN_NAME_DETAILED(prefix, suffix)
#define PERFORMANCE_MKLDNN_NAME(prefix)
#endif

#ifdef PERFORMANCE_MONITORING

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#ifdef PERFORMANCE_MONITORING_USE_TSC
#include <unistd.h>
#endif
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace performance {

  class PreciseTime {
    static const uint64_t clocks_per_second_ = 1000000000;

    uint64_t time_stamp_;

#ifdef PERFORMANCE_MONITORING_USE_TSC
    static double GetTSCFreq() {
      static double tsc_freq = 0;
      if (!tsc_freq) {
        // Calibrate the frequency
        const int usleep_one_second = 1000000;
        uint64_t tsc0 = GetTSC();
        usleep(usleep_one_second);
        uint64_t tsc1 = GetTSC();
        uint64_t tsc_diff = tsc1 - tsc0;
        tsc_freq = (double)tsc_diff / clocks_per_second_;
      }
      return tsc_freq;
    }

    static uint64_t GetTSC() {
      uint32_t lo, hi;
      __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi) : : "%ecx");
      return (uint64_t)lo | ((uint64_t)hi << 32);
    }

    static PreciseTime GetTimeStamp(clockid_t clock_id) {
      return PreciseTime((uint64_t)(GetTSC() / GetTSCFreq()));
    }
#else
    static PreciseTime GetTimeStamp(clockid_t clock_id) {
      timespec current_time;
      clock_gettime(clock_id, &current_time);

      return PreciseTime(clocks_per_second_ * ((uint64_t)current_time.tv_sec)
        + ((uint64_t)current_time.tv_nsec));
    }
#endif

   public:
    PreciseTime() {
    }

    static void Calibrate() {
#ifdef PERFORMANCE_MONITORING_USE_TSC
      GetTSCFreq();
#endif
    }

    explicit PreciseTime(uint64_t time_stamp) : time_stamp_(time_stamp) {
    }

    operator uint64_t() const {
      return time_stamp_;
    }

    PreciseTime& operator=(const uint64_t& time) {
      this->time_stamp_ = time;
      return *this;
    }

    friend PreciseTime operator+(PreciseTime lhs, const PreciseTime& rhs) {
      lhs.time_stamp_ += rhs.time_stamp_;
      return lhs;
    }

    friend PreciseTime operator-(PreciseTime lhs, const PreciseTime& rhs) {
      lhs.time_stamp_ -= rhs.time_stamp_;
      return lhs;
    }

    static PreciseTime GetClocksPerSecond() {
      return PreciseTime(clocks_per_second_);
    }

    static PreciseTime GetMonotonicTime() {
      return GetTimeStamp(CLOCK_MONOTONIC);
    }

    static PreciseTime GetProcessTime() {
      return GetTimeStamp(CLOCK_THREAD_CPUTIME_ID);
    }
  };

  class Measurement {
    PreciseTime process_accumulator_;
    PreciseTime process_time_stamp_;
    PreciseTime monotonic_accumulator_;
    PreciseTime monotonic_time_stamp_;
    Measurement* next_;

    static Measurement*& GetStack() {
      static Measurement* stack = NULL;
      return stack;
    }

    void Suspend() {
      process_accumulator_ = process_accumulator_ +
        PreciseTime::GetProcessTime() - process_time_stamp_;
      monotonic_accumulator_ = monotonic_accumulator_ +
        PreciseTime::GetMonotonicTime() - monotonic_time_stamp_;
    }

    void Resume() {
      monotonic_time_stamp_ = PreciseTime::GetMonotonicTime();
      process_time_stamp_ = PreciseTime::GetProcessTime();
    }

   public:
    Measurement() {
    }

    void Start() {
      static Measurement*& stack = GetStack();

      if (stack)
          stack->Suspend();

      next_ = stack;
      stack = this;

      monotonic_accumulator_ = 0;
      process_accumulator_ = 0;
      monotonic_time_stamp_ = PreciseTime::GetMonotonicTime();
      process_time_stamp_ = PreciseTime::GetProcessTime();
    }

    void Stop() {
      process_accumulator_ = process_accumulator_ +
        PreciseTime::GetProcessTime() - process_time_stamp_;
      monotonic_accumulator_ = monotonic_accumulator_ + 
        PreciseTime::GetMonotonicTime() - monotonic_time_stamp_;

      static Measurement*& stack = GetStack();

      stack = next_;

      if (stack)
          stack->Resume();
    }

    const PreciseTime &GetProcessTimeStamp() const {
      return process_accumulator_;
    }

    const PreciseTime &GetMonotonicTimeStamp() const {
      return monotonic_accumulator_;
    }
  };

  class Event {
    unsigned number_of_calls_;
    PreciseTime total_process_time_;
    PreciseTime minimal_process_time_;
    PreciseTime maximal_process_time_;

    PreciseTime total_monotonic_time_;
    PreciseTime minimal_monotonic_time_;
    PreciseTime maximal_monotonic_time_;

   public:
    Event() : number_of_calls_(0),
              total_process_time_(0),
              minimal_process_time_(0),
              maximal_process_time_(0),
              total_monotonic_time_(0),
              minimal_monotonic_time_(0),
              maximal_monotonic_time_(0) {
    }

    void Update(const Measurement &measurement) {
      const PreciseTime &process_time_stamp = measurement.GetProcessTimeStamp();
      const PreciseTime &monotonic_time_stamp =
        measurement.GetMonotonicTimeStamp();

      total_process_time_ = total_process_time_ + process_time_stamp;
      total_monotonic_time_ = total_monotonic_time_ + monotonic_time_stamp;

      if (minimal_process_time_ > process_time_stamp || !number_of_calls_)
          minimal_process_time_ = process_time_stamp;

      if (maximal_process_time_ < process_time_stamp || !number_of_calls_)
          maximal_process_time_ = process_time_stamp;

      if (minimal_monotonic_time_ > monotonic_time_stamp || !number_of_calls_)
          minimal_monotonic_time_ = monotonic_time_stamp;

      if (maximal_monotonic_time_ < monotonic_time_stamp || !number_of_calls_)
          maximal_monotonic_time_ = monotonic_time_stamp;

      number_of_calls_++;
    }

    PreciseTime GetTotalProcessTime() const {
      return total_process_time_;
    }

    PreciseTime GetAverageMonotonicTime() const {
      return number_of_calls_
        ? PreciseTime(total_monotonic_time_ / number_of_calls_)
        : PreciseTime(0);
    }

    PreciseTime GetMinimalMonotonicTime() const {
      return minimal_monotonic_time_;
    }

    PreciseTime GetMaximalMonotonicTime() const {
      return maximal_monotonic_time_;
    }

    PreciseTime GetAverageProcessTime() const {
      return number_of_calls_
      ? PreciseTime(total_process_time_ / number_of_calls_)
      : PreciseTime(0);
    }

    PreciseTime GetMinimalProcessTime() const {
      return minimal_process_time_;
    }

    PreciseTime GetMaximalProcessTime() const {
      return maximal_process_time_;
    }

    PreciseTime GetNumberOfCalls() const {
      return PreciseTime(number_of_calls_);
    }
  };

  class Log {
   public:
    static void WriteLine() {
      printf("\n");
    }

    static void WriteLine(const char* string) {
      printf("%31s\n", string);
    }

    static void WriteHeaders() {
      printf("%10s %16s %16s %16s %16s %16s %16s : %s\n\n",
        "Calls",
        "Avg(total)", "Min(total)", "Max(total)",
        "Avg(proc)", "Min(proc)", "Max(proc)",
        "Layer");
    }

    static void WriteNoSpacing(const char* string, const PreciseTime& time) {
      printf("%18lu : %s\n", (uint64_t)time, string);
    }

    static void Write(const char* string, const PreciseTime& time) {
      printf("%18lu %10c %s\n", (uint64_t)time, ':', string);
    }

    static void Write(const char* string, const PreciseTime& time,
      double percentage) {
      printf("%18lu %6.2f %% : %s\n", (uint64_t)time, percentage, string);
    }

    static void Write(const char *string, const Event &event) {
      printf("%10lu %16lu %16lu %16lu %16lu %16lu %16lu : %s \n",
          (uint64_t)event.GetNumberOfCalls(),
          (uint64_t)event.GetAverageMonotonicTime(),
          (uint64_t)event.GetMinimalMonotonicTime(),
          (uint64_t)event.GetMaximalMonotonicTime(),
          (uint64_t)event.GetAverageProcessTime(),
          (uint64_t)event.GetMinimalProcessTime(),
          (uint64_t)event.GetMaximalProcessTime(),
          string);
    }
  };

  class Monitor {
    typedef std::vector<std::string> NameVector;
    typedef std::vector<Event> EventVector;
    typedef std::pair<std::string, unsigned> Pair;
    typedef std::map<std::string, unsigned> Map;
    typedef Map::iterator Iterator;
    typedef std::pair<Iterator, bool> Status;

    EventVector events_;
    Map event_name_id_map_;

    bool are_measurements_enabled_;
    bool resetting_;

    NameVector event_names_;
    PreciseTime total_non_mkl_time_;
    PreciseTime total_mkl_time_;
    PreciseTime total_mkl_conversions_time_;
    PreciseTime total_data_layer_time_;
    PreciseTime total_weights_update_time_;
    PreciseTime total_monotonic_time_;
    PreciseTime total_init_time_;
    PreciseTime total_process_time_;

    void DumpStatistics() {
      if (events_.size())
        DumpEventsLog();

      DumpGeneralLog();
    }

    void DumpEventsLog() {
      ObtainEventNames();
      ObtainTotalMklConversionTime();
      ObtainTotalWeightsUpdateTime();
      ObtainTotalDataLayerTime();
      ObtainTotalMklTime();

      Log::WriteLine();
      Log::WriteLine("Detailed event information");
      Log::WriteLine();
      Log::WriteHeaders();
      DumpDetailedEventInformation();
    }

    void DumpGeneralLog() {
      Log::WriteLine();
      Log::WriteLine();
      Log::WriteLine("Total event execution time");
      Log::WriteLine();
      DumpEventTimings();

      Log::WriteLine();
      Log::WriteLine();
      Log::WriteLine("Summarized information");
      Log::WriteLine();
      DumpGeneralTimings();

      Log::WriteLine();
    }

    void DumpGeneralTimings() {
      const PreciseTime framework_time = total_process_time_ -
        total_non_mkl_time_ - total_mkl_time_ -
        total_mkl_conversions_time_ - total_data_layer_time_ -
        total_weights_update_time_ - total_init_time_;
      const PreciseTime system_time = total_monotonic_time_ -
        total_process_time_;

      Log::Write("Data layer", total_data_layer_time_,
        GetTimePercentage(total_data_layer_time_));
      Log::Write("Weight update", total_weights_update_time_,
        GetTimePercentage(total_weights_update_time_));
      Log::Write("Non-MKL(DNN) events", total_non_mkl_time_,
        GetTimePercentage(total_non_mkl_time_));
      Log::Write("MKL(DNN) conversions", total_mkl_conversions_time_,
        GetTimePercentage(total_mkl_conversions_time_));
      Log::Write("MKL(DNN) events", total_mkl_time_,
        GetTimePercentage(total_mkl_time_));
      Log::Write("Framework", framework_time,
        GetTimePercentage(framework_time));
      Log::Write("System", system_time, GetTimePercentage(system_time));
      Log::Write("Initialization", total_init_time_);
      Log::Write("Process", total_process_time_);
      Log::Write("Total", total_monotonic_time_);
    }

    void DumpEventTimings() {
      for (unsigned i = 0; i < events_.size(); i++) {
        Log::WriteNoSpacing(event_names_[i].c_str(),
          events_[i].GetTotalProcessTime());
      }
    }

    void DumpDetailedEventInformation() {
      for (unsigned i = 0; i < events_.size(); i++) {
        Log::Write(event_names_[i].c_str(), events_[i]);
      }
    }

    void ObtainTotalMklTime() {
      total_non_mkl_time_ = 0;
      total_mkl_time_ = 0;

      Iterator iterator = event_name_id_map_.begin();
      for (; iterator != event_name_id_map_.end(); iterator++) {
        if (iterator->first.find("mkl") != std::string::npos)
          total_mkl_time_ = total_mkl_time_ +
            events_[iterator->second].GetTotalProcessTime();
        else
          total_non_mkl_time_ = total_non_mkl_time_ +
            events_[iterator->second].GetTotalProcessTime();
      }
      total_non_mkl_time_ = total_non_mkl_time_ -
        total_weights_update_time_ - total_data_layer_time_;
      total_mkl_time_ = total_mkl_time_ - total_mkl_conversions_time_;
    }

    void ObtainTotalMklConversionTime() {
      if (event_name_id_map_.count("mkl_conversion") > 0) {
        unsigned mkl_conv_id = event_name_id_map_["mkl_conversion"];
        total_mkl_conversions_time_ =
          events_[mkl_conv_id].GetTotalProcessTime();
      } else if (event_name_id_map_.count("mkldnn_conversion") > 0) {
        unsigned mkldnn_conv_id = event_name_id_map_["mkldnn_conversion"];
        total_mkl_conversions_time_ =
          events_[mkldnn_conv_id].GetTotalProcessTime();
      } else {
        total_mkl_conversions_time_ = 0;
      }
    }

    void ObtainTotalDataLayerTime() {
      for (unsigned i = 0; i < event_names_.size(); i++) {
        if (event_names_[i].find("W_") != std::string::npos) {
          total_data_layer_time_ = events_[i].GetTotalProcessTime();
          break;
        }
      }
    }

    void ObtainTotalWeightsUpdateTime() {
      if (event_name_id_map_.count("weights_update") > 0) {
        unsigned weights_update_id = event_name_id_map_["weights_update"];
        total_weights_update_time_ =
          events_[weights_update_id].GetTotalProcessTime();
      } else {
        total_weights_update_time_ = 0;
      }
    }

    void ObtainEventNames() {
      event_names_.resize(event_name_id_map_.size());

      Iterator iterator = event_name_id_map_.begin();
      for (; iterator != event_name_id_map_.end(); iterator++)
        event_names_[iterator->second] = iterator->first;
    }

    double GetTimePercentage(const PreciseTime& time) {
      return (100.0 * static_cast<double>(time)) /
        static_cast<double>(total_monotonic_time_ - total_init_time_);
    }

   public:
    Monitor() {
      resetting_ = false;
      events_.reserve(64);

      PreciseTime::Calibrate();

      are_measurements_enabled_ = false;

      total_monotonic_time_ = PreciseTime::GetMonotonicTime();
      total_process_time_ = PreciseTime::GetProcessTime();
      total_init_time_ = 0;
    }

    ~Monitor() {
      total_process_time_ = PreciseTime::GetProcessTime() - total_process_time_;
      total_monotonic_time_ = PreciseTime::GetMonotonicTime() -
        total_monotonic_time_;

      if (are_measurements_enabled_)
        DumpStatistics();
    }

    void EnableMeasurements() {
      are_measurements_enabled_ = true;
    }

    void MarkAsInitialized() {
      total_init_time_ = PreciseTime::GetProcessTime() - total_process_time_;
    }

    void StartResetting() { resetting_ = true; }
    void StopResetting() { resetting_ = false; }

    unsigned GetEventIdByName(const char *event_name) {
      if (!are_measurements_enabled_)
        return PERFORMANCE_EVENT_ID_UNSET;

      Pair pair(event_name, events_.size());
      Status status = event_name_id_map_.insert(pair);

      // If insertion succeeded
      if (status.second)
        events_.push_back(Event());

      return status.first->second;
    }

    void UpdateEventById(unsigned event_id, const Measurement &measurement) {
      if (are_measurements_enabled_) {
        if (resetting_)
          events_[event_id] = Event();
        else
          events_[event_id].Update(measurement);
      }
    }
  };

  extern Monitor monitor;

}  // namespace performance

#endif  // ifdef PERFORMANCE_MONITORING
#endif  // ifndef PerformanceH
