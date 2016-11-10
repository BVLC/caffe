#ifndef PerformanceH
#define PerformanceH

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>
#include <map>

// #include "caffe/util/cpu_info.hpp"


namespace Performance {

	class PreciseTime
	{
		static const uint64_t precision = 1000000000;

		uint64_t timeStamp;

		static PreciseTime getTimeStamp(clockid_t clockId)
		{
			timespec currentTime;
			clock_gettime(clockId, &currentTime);

			return precision * ((uint64_t)currentTime.tv_sec) + ((uint64_t)currentTime.tv_nsec);
		}

	public:

		PreciseTime()
		{
		}

		PreciseTime(uint64_t timeStamp) : timeStamp(timeStamp)
		{
		}

		operator uint64_t () const
		{
			return timeStamp;
		}

		static PreciseTime getPrecision()
		{
			return precision;
		}

		static PreciseTime getMonotonicTime()
		{
			return getTimeStamp(CLOCK_MONOTONIC);
		}

		static PreciseTime getProcessTime()
		{
			return getTimeStamp(CLOCK_PROCESS_CPUTIME_ID);
		}
	};

	class Measurement
	{
		PreciseTime processTimeStamp;
		PreciseTime systemTimeStamp;

	public:

		Measurement()
		{
		}

		void start()
		{
			processTimeStamp = PreciseTime::getProcessTime();
			systemTimeStamp = PreciseTime::getMonotonicTime();
		}

		void stop()
		{
			processTimeStamp = PreciseTime::getProcessTime() - processTimeStamp;
			const uint64_t monotonicTime = PreciseTime::getMonotonicTime();

			// TODO: Make these variables static fields of this class.
			// caffe::cpu::CpuInfo cpuInfo;
			// caffe::cpu::Collection collection(&cpuInfo);
			// printf("%d - num of cpu cores ", collection.getTotalNumberOfCpuCores());
			
			const unsigned numberOfThreads = 44; // TEMP, TODO: Replace this value with call to Collection::getTotalNumberOfCpuCores().
			const uint64_t realProcessTime = processTimeStamp / numberOfThreads;
			
			processTimeStamp = realProcessTime;
			
			systemTimeStamp = monotonicTime - systemTimeStamp;
		}

		const PreciseTime &getProcessTimeStamp() const
		{
			return processTimeStamp;
		}

		const PreciseTime &getSystemTimeStamp() const
		{
			return systemTimeStamp;
		}
	};

	class Event
	{
		unsigned numberOfCalls;
		PreciseTime totalProcessTime;
		PreciseTime minimalProcessTime;
		PreciseTime maximalProcessTime;
		
		PreciseTime totalSystemTime;
		PreciseTime minimalSystemTime;
		PreciseTime maximalSystemTime;

	public:

		Event() : numberOfCalls(0), totalProcessTime(0), minimalProcessTime(0), maximalProcessTime(0), totalSystemTime(0), minimalSystemTime(0), maximalSystemTime(0)
		{
		}

		void update(const Measurement &measurement)
		{
			const PreciseTime &timeStamp = measurement.getProcessTimeStamp();
			const PreciseTime &systemTimeStamp = measurement.getSystemTimeStamp();

			totalProcessTime = totalProcessTime + timeStamp;
			totalSystemTime = totalSystemTime + systemTimeStamp;
			
			if (minimalProcessTime > timeStamp || !numberOfCalls)
				minimalProcessTime = timeStamp;

			if (maximalProcessTime < timeStamp || !numberOfCalls)
				maximalProcessTime = timeStamp;
				
			if (minimalSystemTime > systemTimeStamp || !numberOfCalls)
				minimalSystemTime = systemTimeStamp;

			if (maximalSystemTime < systemTimeStamp || !numberOfCalls)
				maximalSystemTime = systemTimeStamp;

			numberOfCalls++;
		}

		PreciseTime getTotalTime() const
		{
			return totalProcessTime;
		}

		PreciseTime getAverageSystemTime() const
		{
			return numberOfCalls ? totalSystemTime / numberOfCalls : 0;
		}

		PreciseTime getMinimalSystemTime() const
		{
			return minimalSystemTime;
		}

		PreciseTime getMaximalSystemTime() const
		{
			return maximalSystemTime;
		}
		
		PreciseTime getAverageTime() const
		{
			return numberOfCalls ? totalProcessTime / numberOfCalls : 0;
		}

		PreciseTime getMinimalTime() const
		{
			return minimalProcessTime;
		}

		PreciseTime getMaximalTime() const
		{
			return maximalProcessTime;
		}

		PreciseTime getNumberOfCalls() const
		{
			return numberOfCalls;
		}
	};

	class Log
	{
	public:

		static void writeLine()
		{
			printf("\n");
		}

		static void writeHeaders()
		{
			printf("\n                       Avg(proc)     Min(proc)     Max(proc)         Avg(total)             Min(total)             Max(total)        Calls\n\n");
		}

		static void write(const char *string, const PreciseTime &time)
		{
			printf("%16s : %lu [nsec]\n", string, (uint64_t)time);
		}

		static void write(const char *string, const Event &event)
		{
			printf("%16s : %12lu, %12lu, %12lu, %16lu, %20lu, %20lu, %12lu\n", string,
				(uint64_t)event.getAverageTime(),
				(uint64_t)event.getMinimalTime(),
				(uint64_t)event.getMaximalTime(),
				(uint64_t)event.getAverageSystemTime(),
				(uint64_t)event.getMinimalSystemTime(),
				(uint64_t)event.getMaximalSystemTime(),
				(uint64_t)event.getNumberOfCalls());
		}
	};

	class Monitor
	{
		typedef std::vector<Event> Vector;
		typedef std::pair<std::string, unsigned> Pair;
		typedef std::map<std::string, unsigned> Map;
		typedef Map::iterator Iterator;
		typedef std::pair<Iterator, bool> Status;

		Vector events;
		Map eventNameIdMap;

		bool isMeasurementsEnabled;

		PreciseTime totalMonotonicTime;
		PreciseTime totalProcessTime;
		PreciseTime totalEventsTime;
		PreciseTime totalInitTime;

		void dumpStatistics()
		{
			if (events.size())
				dumpEventsLog();

			dumpGeneralLog();
		}

		void dumpEventsLog()
		{
			obtainTotalEventsTime();

			Log::writeHeaders();

			dumpDetailedEventInformation();
		}

		void dumpGeneralLog()
		{
			Log::writeLine();

			dumpEventTimings();

			Log::writeLine();

			dumpGeneralTimings();

			Log::writeLine();
		}

		void dumpGeneralTimings()
		{
			Log::write("Initialization", totalInitTime);
			Log::write("Framework", totalProcessTime - totalEventsTime - totalInitTime);
			Log::write("System", totalMonotonicTime - totalProcessTime);
			Log::write("Process", totalProcessTime);
			Log::write("All events", totalEventsTime);
			Log::write("Total", totalMonotonicTime);
		}

		void dumpEventTimings()
		{
			Iterator iterator = eventNameIdMap.begin();
			for (; iterator != eventNameIdMap.end(); iterator++)
				Log::write(iterator->first.c_str(), events[iterator->second].getTotalTime());
		}

		void dumpDetailedEventInformation()
		{
			Iterator iterator = eventNameIdMap.begin();
			for (; iterator != eventNameIdMap.end(); iterator++)
				Log::write(iterator->first.c_str(), events[iterator->second]);
		}

		void obtainTotalEventsTime()
		{
			totalEventsTime = 0;
			Iterator iterator = eventNameIdMap.begin();
			for (; iterator != eventNameIdMap.end(); iterator++)
				totalEventsTime = totalEventsTime + events[iterator->second].getTotalTime();
		}

	public:

		Monitor()
		{
			events.reserve(64);

			isMeasurementsEnabled = false;

			totalMonotonicTime = PreciseTime::getMonotonicTime();
			totalProcessTime = PreciseTime::getProcessTime();
			totalEventsTime = 0;
			totalInitTime = 0;
		}

		~Monitor()
		{
			totalProcessTime = PreciseTime::getProcessTime() - totalProcessTime;
			totalMonotonicTime = PreciseTime::getMonotonicTime() - totalMonotonicTime;

			if (isMeasurementsEnabled)
				dumpStatistics();
		}

		void enableMeasurements()
		{
			isMeasurementsEnabled = true;
		}

		void markAsInitialized()
		{
			totalInitTime = PreciseTime::getProcessTime() - totalProcessTime;
		}

		unsigned getEventIdByName(const char *eventName)
		{
			if (!isMeasurementsEnabled)
				return 0;

			Pair pair(eventName, events.size());
			Status status = eventNameIdMap.insert(pair);

			// If insertion succeeded
			if (status.second)
				events.push_back(Event());

			return status.first->second;
		}

		void updateEventById(unsigned eventId, const Measurement &measurement)
		{
			if (isMeasurementsEnabled)
				events[eventId].update(measurement);
		}
	};
	 
	extern Monitor monitor;

}

#endif
