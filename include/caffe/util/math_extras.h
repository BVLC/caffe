#ifndef __MATH_EXTRAS_H__
#define __MATH_EXTRAS_H__

#ifdef _MSC_VER
# define snprintf _snprintf

// VS2013 has most of the math functions now, but we still need to work
// around various differences in behavior of Inf.
#if _MSC_VER < 1800

namespace std {
  inline bool signbit(float num);
  inline bool signbit(double num);
} // namespace std

#endif // _MSC_VER < 1800
#endif // _MSC_VER

#endif //__MATH_EXTRAS_H__