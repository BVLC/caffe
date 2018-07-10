#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_rng_uniform<half_fp>(const int_tp n,
                             const half_fp a, const half_fp b,
                             half_fp* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<float> random_distribution(static_cast<float>(a),
    caffe_nextafter<float>(static_cast<float>(b)));

  boost::variate_generator<caffe::rng_t*,
    boost::uniform_real<float>> variate_generator(
    caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<half_fp>(const int_tp n,
                         const half_fp a, const half_fp sigma,
                         half_fp* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float> > variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<uint8_t>(const int_tp n, const uint8_t a,
                                 const uint8_t sigma, uint8_t* r) {
  // FIXME
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float> > variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<uint16_t>(const int_tp n, const uint16_t a,
                                  const uint16_t sigma, uint16_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float>> variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<uint32_t>(const int_tp n, const uint32_t a,
                                  const uint32_t sigma, uint32_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float>> variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<uint64_t>(const int_tp n, const uint64_t a,
                                  const uint64_t sigma, uint64_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float>> variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_bernoulli<half_fp, int32_t>(const int_tp n,
                                    const half_fp p, int32_t* r) {
  // FIXME
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  float f_p = p;
  boost::bernoulli_distribution<float> random_distribution(f_p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<float> > variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<int32_t>(variate_generator());
  }
}

template<>
void caffe_rng_bernoulli<half_fp, uint32_t>(const int_tp n,
                                             const half_fp p, uint32_t* r) {
  // FIXME
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  float f_p = p;
  boost::bernoulli_distribution<float> random_distribution(f_p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<float>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<uint32_t>(variate_generator());
  }
}

uint_tp caffe_rng_rand() {
  return (*caffe_rng())();
}

template<typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template<>
void caffe_rng_uniform<uint8_t>(const uint_tp n, uint8_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint8_t> random_distribution(
      std::numeric_limits<uint8_t>::min(),
      std::numeric_limits<uint8_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint8_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_uniform<uint16_t>(const uint_tp n, uint16_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint16_t> random_distribution(
      std::numeric_limits<uint16_t>::min(),
      std::numeric_limits<uint16_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint16_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_uniform<uint32_t>(const uint_tp n, uint32_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint32_t> random_distribution(
      std::numeric_limits<uint32_t>::min(),
      std::numeric_limits<uint32_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint32_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_uniform<uint64_t>(const uint_tp n, uint64_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint64_t> random_distribution(
      std::numeric_limits<uint64_t>::min(),
      std::numeric_limits<uint64_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint64_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<typename Dtype>
void caffe_rng_uniform(const int_tp n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*,
  boost::uniform_real<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void caffe_rng_uniform<float>(const int_tp n, const float a,
                                       const float b, float* r);
template void caffe_rng_uniform<double>(const int_tp n, const double a,
                                        const double b, double* r);
template void caffe_rng_uniform<uint8_t>(const int_tp n, const uint8_t a,
                                         const uint8_t b, uint8_t* r);
template void caffe_rng_uniform<uint16_t>(const int_tp n, const uint16_t a,
                                          const uint16_t b, uint16_t* r);
template void caffe_rng_uniform<uint32_t>(const int_tp n, const uint32_t a,
                                          const uint32_t b, uint32_t* r);
template void caffe_rng_uniform<uint64_t>(const int_tp n, const uint64_t a,
                                          const uint64_t b, uint64_t* r);


template<typename Dtype>
void caffe_rng_gaussian(const int_tp n, const Dtype a, const Dtype sigma,
                        Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*,
  boost::normal_distribution<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int_tp n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int_tp n, const double mu,
                                const double sigma, double* r);

template<typename Dtype, typename Itype>
void caffe_rng_bernoulli(const int_tp n, const Dtype p, Itype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<Itype>(variate_generator());
  }
}

template void caffe_rng_bernoulli<half_fp, int8_t>(const int_tp n,
                                  const half_fp p, int8_t* r);
template void caffe_rng_bernoulli<half_fp, int16_t>(const int_tp n,
                                  const half_fp p, int16_t* r);
template void caffe_rng_bernoulli<half_fp, int32_t>(const int_tp n,
                                  const half_fp p, int32_t* r);
template void caffe_rng_bernoulli<half_fp, int64_t>(const int_tp n,
                                  const half_fp p, int64_t* r);
template void caffe_rng_bernoulli<half_fp, uint8_t>(const int_tp n,
                                  const half_fp p, uint8_t* r);
template void caffe_rng_bernoulli<half_fp, uint16_t>(const int_tp n,
                                  const half_fp p, uint16_t* r);
template void caffe_rng_bernoulli<half_fp, uint32_t>(const int_tp n,
                                  const half_fp p, uint32_t* r);
template void caffe_rng_bernoulli<half_fp, uint64_t>(const int_tp n,
                                  const half_fp p, uint64_t* r);

template void caffe_rng_bernoulli<float, int8_t>(const int_tp n,
                                  const float p, int8_t* r);
template void caffe_rng_bernoulli<float, int16_t>(const int_tp n,
                                  const float p, int16_t* r);
template void caffe_rng_bernoulli<float, int32_t>(const int_tp n,
                                  const float p, int32_t* r);
template void caffe_rng_bernoulli<float, int64_t>(const int_tp n,
                                  const float p, int64_t* r);
template void caffe_rng_bernoulli<float, uint8_t>(const int_tp n,
                                  const float p, uint8_t* r);
template void caffe_rng_bernoulli<float, uint16_t>(const int_tp n,
                                  const float p, uint16_t* r);
template void caffe_rng_bernoulli<float, uint32_t>(const int_tp n,
                                  const float p, uint32_t* r);
template void caffe_rng_bernoulli<float, uint64_t>(const int_tp n,
                                  const float p, uint64_t* r);

template void caffe_rng_bernoulli<double, int8_t>(const int_tp n,
                                  const double p, int8_t* r);
template void caffe_rng_bernoulli<double, int16_t>(const int_tp n,
                                  const double p, int16_t* r);
template void caffe_rng_bernoulli<double, int32_t>(const int_tp n,
                                  const double p, int32_t* r);
template void caffe_rng_bernoulli<double, int64_t>(const int_tp n,
                                  const double p, int64_t* r);
template void caffe_rng_bernoulli<double, uint8_t>(const int_tp n,
                                  const double p, uint8_t* r);
template void caffe_rng_bernoulli<double, uint16_t>(const int_tp n,
                                  const double p, uint16_t* r);
template void caffe_rng_bernoulli<double, uint32_t>(const int_tp n,
                                  const double p, uint32_t* r);
template void caffe_rng_bernoulli<double, uint64_t>(const int_tp n,
                                  const double p, uint64_t* r);

template void caffe_rng_bernoulli<uint8_t, int8_t>(const int_tp n,
                                  const uint8_t p, int8_t* r);
template void caffe_rng_bernoulli<uint8_t, int16_t>(const int_tp n,
                                  const uint8_t p, int16_t* r);
template void caffe_rng_bernoulli<uint8_t, int32_t>(const int_tp n,
                                  const uint8_t p, int32_t* r);
template void caffe_rng_bernoulli<uint8_t, int64_t>(const int_tp n,
                                  const uint8_t p, int64_t* r);
template void caffe_rng_bernoulli<uint8_t, uint8_t>(const int_tp n,
                                  const uint8_t p, uint8_t* r);
template void caffe_rng_bernoulli<uint8_t, uint16_t>(const int_tp n,
                                  const uint8_t p, uint16_t* r);
template void caffe_rng_bernoulli<uint8_t, uint32_t>(const int_tp n,
                                  const uint8_t p, uint32_t* r);
template void caffe_rng_bernoulli<uint8_t, uint64_t>(const int_tp n,
                                  const uint8_t p, uint64_t* r);

template void caffe_rng_bernoulli<uint16_t, int8_t>(const int_tp n,
                                  const uint16_t p, int8_t* r);
template void caffe_rng_bernoulli<uint16_t, int16_t>(const int_tp n,
                                  const uint16_t p, int16_t* r);
template void caffe_rng_bernoulli<uint16_t, int32_t>(const int_tp n,
                                  const uint16_t p, int32_t* r);
template void caffe_rng_bernoulli<uint16_t, int64_t>(const int_tp n,
                                  const uint16_t p, int64_t* r);
template void caffe_rng_bernoulli<uint16_t, uint8_t>(const int_tp n,
                                  const uint16_t p, uint8_t* r);
template void caffe_rng_bernoulli<uint16_t, uint16_t>(const int_tp n,
                                  const uint16_t p, uint16_t* r);
template void caffe_rng_bernoulli<uint16_t, uint32_t>(const int_tp n,
                                  const uint16_t p, uint32_t* r);
template void caffe_rng_bernoulli<uint16_t, uint64_t>(const int_tp n,
                                  const uint16_t p, uint64_t* r);

template void caffe_rng_bernoulli<uint32_t, int8_t>(const int_tp n,
                                  const uint32_t p, int8_t* r);
template void caffe_rng_bernoulli<uint32_t, int16_t>(const int_tp n,
                                  const uint32_t p, int16_t* r);
template void caffe_rng_bernoulli<uint32_t, int32_t>(const int_tp n,
                                  const uint32_t p, int32_t* r);
template void caffe_rng_bernoulli<uint32_t, int64_t>(const int_tp n,
                                  const uint32_t p, int64_t* r);
template void caffe_rng_bernoulli<uint32_t, uint8_t>(const int_tp n,
                                  const uint32_t p, uint8_t* r);
template void caffe_rng_bernoulli<uint32_t, uint16_t>(const int_tp n,
                                  const uint32_t p, uint16_t* r);
template void caffe_rng_bernoulli<uint32_t, uint32_t>(const int_tp n,
                                  const uint32_t p, uint32_t* r);
template void caffe_rng_bernoulli<uint32_t, uint64_t>(const int_tp n,
                                  const uint32_t p, uint64_t* r);

template void caffe_rng_bernoulli<uint64_t, int8_t>(const int_tp n,
                                  const uint64_t p, int8_t* r);
template void caffe_rng_bernoulli<uint64_t, int16_t>(const int_tp n,
                                  const uint64_t p, int16_t* r);
template void caffe_rng_bernoulli<uint64_t, int32_t>(const int_tp n,
                                  const uint64_t p, int32_t* r);
template void caffe_rng_bernoulli<uint64_t, int64_t>(const int_tp n,
                                  const uint64_t p, int64_t* r);
template void caffe_rng_bernoulli<uint64_t, uint8_t>(const int_tp n,
                                  const uint64_t p, uint8_t* r);
template void caffe_rng_bernoulli<uint64_t, uint16_t>(const int_tp n,
                                  const uint64_t p, uint16_t* r);
template void caffe_rng_bernoulli<uint64_t, uint32_t>(const int_tp n,
                                  const uint64_t p, uint32_t* r);
template void caffe_rng_bernoulli<uint64_t, uint64_t>(const int_tp n,
                                  const uint64_t p, uint64_t* r);

}  // namespace caffe
