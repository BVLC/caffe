/*  This file is part of Structured Prediction (SP) - http://www.alexander-schwing.de/
 *
 *  Structured Prediction (SP) is free software: you can
 *  redistribute it and/or modify it under the terms of the GNU General
 *  Public License as published by the Free Software Foundation, either
 *  version 3 of the License, or (at your option) any later version.
 *
 *  Structured Prediction (SP) is distributed in the hope
 *  that it will be useful, but WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE. See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Structured Prediction (SP).
 *  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Copyright (C) 2010-2013  Alexander G. Schwing  [http://www.alexander-schwing.de/]
 */

//Author: Alexander G. Schwing

/*  ON A PERSONAL NOTE: I spent a significant amount of time to go through
 *  both, theoretical justifications and coding of this framework.
 *  I hope the package is useful for your task. Any citations, requests, 
 *  feedback, donations and support would be greatly appreciated.
 *  Thank you for contacting me!
 */

#ifdef USE_ON_WINDOWS
#include <windows.h>
#else
#include <sys/time.h>
#endif

class CPrecisionTimer
{
#ifdef USE_ON_WINDOWS
	LARGE_INTEGER lFreq, lStart;
#else
	timeval lStart;
#endif

public:
	CPrecisionTimer()
	{
#ifdef USE_ON_WINDOWS
		QueryPerformanceFrequency(&lFreq);
#endif
	}

	inline void Start()
	{
#ifdef USE_ON_WINDOWS
		QueryPerformanceCounter(&lStart);
#else
		gettimeofday(&lStart, 0);
#endif
	}

	inline double Stop()
	{
		// Return duration in seconds...
#ifdef USE_ON_WINDOWS
		LARGE_INTEGER lEnd;
		QueryPerformanceCounter(&lEnd);
		return (double(lEnd.QuadPart - lStart.QuadPart) / lFreq.QuadPart);
#else
		timeval lFinish;
		gettimeofday(&lFinish, 0);
		return double(lFinish.tv_sec - lStart.tv_sec) + double(lFinish.tv_usec - lStart.tv_usec)/1e6; 
#endif
	}
};