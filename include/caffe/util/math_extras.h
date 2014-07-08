#ifndef __MATH_EXTRAS_H__
#define __MATH_EXTRAS_H__

//#include <math_functions.h>

#ifdef _MSC_VER
// VS2013 has most of the math functions now, but we still need to work
// around various differences in behavior of Inf.
#if _MSC_VER < 1800


namespace std {
	inline bool isinf(double num) { return !_finite(num) && !_isnan(num); }
	inline bool isnan(double num) { return !!_isnan(num); }
	inline bool isfinite(double x) { return _finite(x); }
	__device__ __host__ inline bool signbit(float num) { return _copysign(1.0f, num) < 0; }
	__device__ __host__ inline bool signbit(double num) { return _copysign(1.0, num) < 0; }
} // namespace std

//inline double nextafter(double x, double y) { return _nextafter(x, y); }
//inline float nextafterf(float x, float y) { return x > y ? x - FLT_EPSILON : x + FLT_EPSILON; }
//inline double copysign(double x, double y) { return _copysign(x, y); }
#endif // _MSC_VER
#endif

#endif //__MATH_EXTRAS_H__