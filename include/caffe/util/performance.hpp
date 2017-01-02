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
#define PERFORMANCE_MEASUREMENT_BEGIN()            \
  performance::Measurement m_MACRO;                \
  m_MACRO.Start();

#define PERFORMANCE_MEASUREMENT_END(name)                        \
  m_MACRO.Stop();                                                \
  const char* n_MACRO = name;                                    \
  int id_MACRO = performance::monitor.GetEventIdByName(n_MACRO); \
  performance::monitor.UpdateEventById(id_MACRO, m_MACRO);

#define PERFORMANCE_MEASUREMENT_END_STATIC(name)                        \
  m_MACRO.Stop();                                                       \
  static const char* n_MACRO = name;                                    \
  static int id_MACRO = performance::monitor.GetEventIdByName(n_MACRO); \
  performance::monitor.UpdateEventById(id_MACRO, m_MACRO);

#define PERFORMANCE_CREATE_MONITOR() \
  namespace performance {            \
  Monitor monitor; };

#define PERFORMANCE_INIT_MONITOR()           \
  performance::monitor.EnableMeasurements(); \
  performance::monitor.MarkAsInitialized();

#define PERFORMANCE_MKL_NAME_SFX(prefix, suffix)             \
  (std::string(prefix) + "_mkl_" + this->layer_param_.name() \
    + std::string(suffix)).c_str();

#define PERFORMANCE_MKL_NAME(prefix) \
  (std::string(prefix) + "_mkl_" + this->layer_param_.name()).c_str();

#else
#define PERFORMANCE_MEASUREMENT_BEGIN()
#define PERFORMANCE_MEASUREMENT_END(name)
#define PERFORMANCE_MEASUREMENT_END_STATIC(name)
#define PERFORMANCE_CREATE_MONITOR()
#define PERFORMANCE_INIT_MONITOR()
#define PERFORMANCE_MKL_NAME_SFX(prefix, suffix)
#define PERFORMANCE_MKL_NAME(prefix)
#endif

#ifdef PERFORMANCE_MONITORING
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace performance {

  class PreciseTime {
    static const uint64_t clocks_per_second_ = 1000000000;

    uint64_t time_stamp_;

    static PreciseTime GetTimeStamp(clockid_t clock_id) {
      timespec current_time;
      clock_gettime(clock_id, &current_time);

      return PreciseTime(clocks_per_second_ * ((uint64_t)current_time.tv_sec)
        + ((uint64_t)current_time.tv_nsec));
    }

   public:
    PreciseTime() {
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
      process_accumulator_ = PreciseTime::GetProcessTime()
        - process_time_stamp_;
      monotonic_accumulator_ = PreciseTime::GetMonotonicTime()
        - monotonic_time_stamp_;
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
      process_accumulator_ = PreciseTime::GetProcessTime()
        - process_time_stamp_;
      monotonic_accumulator_ = PreciseTime::GetMonotonicTime()
        - monotonic_time_stamp_;

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

    static void WriteHeaders() {
      printf("\n %30s %20s %18s %18s %18s %18s %18s %18s\n\n",
        "Layer", "Calls",
        "Avg(total)", "Min(total)", "Max(total)",
        "Avg(proc)", "Min(proc)", "Max(proc)");
    }

    static void Write(const char *string, const PreciseTime &time) {
      printf("%31s : %lu [nsec]\n", string, (uint64_t)time);
    }

    static void Write(const char *string, const Event &event) {
      printf("%31s : %18lu %18lu %18lu %18lu %18lu %18lu %18lu\n", string,
          (uint64_t)event.GetNumberOfCalls(),
          (uint64_t)event.GetAverageMonotonicTime(),
          (uint64_t)event.GetMinimalMonotonicTime(),
          (uint64_t)event.GetMaximalMonotonicTime(),
          (uint64_t)event.GetAverageProcessTime(),
          (uint64_t)event.GetMinimalProcessTime(),
          (uint64_t)event.GetMaximalProcessTime());
    }
  };

  class Monitor {
    typedef std::vector<Event> Vector;
    typedef std::pair<std::string, unsigned> Pair;
    typedef std::map<std::string, unsigned> Map;
    typedef Map::iterator Iterator;
    typedef std::pair<Iterator, bool> Status;

    Vector events_;
    Map event_name_id_map_;

    bool are_measurements_enabled_;

    PreciseTime total_monotonic_time_;
    PreciseTime total_process_time_;
    PreciseTime total_events_time_;
    PreciseTime total_init_time_;

    void DumpStatistics() {
      if (events_.size())
        DumpEventsLog();

      DumpGeneralLog();
    }

    void DumpEventsLog() {
      ObtainTotalEventsTime();

      Log::WriteHeaders();

      DumpDetailedEventInformation();
    }

    void DumpGeneralLog() {
      Log::WriteLine();

      DumpEventTimings();

      Log::WriteLine();

      DumpGeneralTimings();

      Log::WriteLine();
    }

    void DumpGeneralTimings() {
      Log::Write("Initialization", total_init_time_);
      Log::Write("Framework", total_process_time_ -
        total_events_time_ - total_init_time_);
      Log::Write("System", total_monotonic_time_ - total_process_time_);
      Log::Write("Process", total_process_time_);
      Log::Write("All events", total_events_time_);
      Log::Write("Total", total_monotonic_time_);
    }

    void DumpEventTimings() {
      Iterator iterator = event_name_id_map_.begin();
      for (; iterator != event_name_id_map_.end(); iterator++)
        Log::Write(iterator->first.c_str(),
          events_[iterator->second].GetTotalProcessTime());
    }

    void DumpDetailedEventInformation() {
      Iterator iterator = event_name_id_map_.begin();
      for (; iterator != event_name_id_map_.end(); iterator++)
        Log::Write(iterator->first.c_str(), events_[iterator->second]);
    }

    void ObtainTotalEventsTime() {
      total_events_time_ = 0;
      Iterator iterator = event_name_id_map_.begin();
      for (; iterator != event_name_id_map_.end(); iterator++)
        total_events_time_ = total_events_time_ +
          events_[iterator->second].GetTotalProcessTime();
    }

   public:
    Monitor() {
      events_.reserve(64);

      are_measurements_enabled_ = false;

      total_monotonic_time_ = PreciseTime::GetMonotonicTime();
      total_process_time_ = PreciseTime::GetProcessTime();
      total_events_time_ = 0;
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

    unsigned GetEventIdByName(const char *event_name) {
      if (!are_measurements_enabled_)
        return 0;

      Pair pair(event_name, events_.size());
      Status status = event_name_id_map_.insert(pair);

      // If insertion succeeded
      if (status.second)
        events_.push_back(Event());

      return status.first->second;
    }

    void UpdateEventById(unsigned event_id, const Measurement &measurement) {
      if (are_measurements_enabled_)
        events_[event_id].Update(measurement);
    }
  };

  extern Monitor monitor;

}  // namespace performance

#endif  // ifdef PERFORMANCE_MONITORING
#endif  // ifndef PerformanceH
