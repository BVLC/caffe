#ifndef __MATH_EXTRAS_H__
#define __MATH_EXTRAS_H__

#ifdef _MSC_VER
# define snprintf _snprintf

// VS2013 has most of the math functions now, but we still need to work
// around various differences in behavior of Inf.
#if _MSC_VER < 1800

namespace std {
  inline bool signbit(float num) { return _copysign(1.0f, num) < 0; }
  inline bool signbit(double num) { return _copysign(1.0, num) < 0; }
} // namespace std

#endif // _MSC_VER < 1800
#endif // _MSC_VER

#endif //__MATH_EXTRAS_H__