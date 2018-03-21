#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/util/print_tools.hpp"


namespace caffe {

template <typename TypeParam>
class QuantBlasTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(QuantBlasTest, TestDtypesIntegerAndDevices);



TYPED_TEST(QuantBlasTest, TestGemmComparativeFloatQuant) {
  typedef typename TypeParam::Dtype Dtype;

  // Expect at most 4% error
  float percentile_eps = 0.04;

  std::random_device rdev;
  std::mt19937 rngen(rdev());

  // Need to test > 64 dimension
  std::uniform_int_distribution<int_tp> dimsRand(1, 256);
  std::uniform_int_distribution<int_tp> boolRand(0, 1);
  std::uniform_int_distribution<int_tp> factorRand(-25, 25);
  std::uniform_real_distribution<float> valRand(-2.0, 2.0);


  for (int_tp testIdx = 0; testIdx < 25; ++testIdx) {
    int_tp M = dimsRand(rngen);
    int_tp N = dimsRand(rngen);
    int_tp K = dimsRand(rngen);

    CBLAS_TRANSPOSE trans_A = boolRand(rngen) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = boolRand(rngen) ? CblasTrans : CblasNoTrans;

    bool has_alpha = true;//boolRand(rngen);
    bool has_beta = false;//has_alpha ? boolRand(rngen) : true;

    bool alpha_with_quant = false;//boolRand(rngen) && has_alpha;
    bool beta_with_quant = boolRand(rngen) && has_beta;

    float alpha_val;
    float beta_val;

    if (has_alpha) {
      alpha_val = alpha_with_quant ? valRand(rngen) : float(1.0);
    } else {
      alpha_val = 0.0;
    }

    if (has_beta) {
      beta_val = beta_with_quant ? valRand(rngen) : float(1.0);
    } else {
      beta_val = 0.0;
    }

    vector<int_tp> A_shape(4, 1);
    vector<int_tp> B_shape(4, 1);
    vector<int_tp> C_shape(4, 1);

    A_shape[2] = M;
    A_shape[3] = K;
    B_shape[2] = K;
    B_shape[3] = N;
    C_shape[2] = M;
    C_shape[3] = N;

    Blob<float> A(A_shape, Caffe::GetDefaultDevice());
    Blob<float> B(B_shape, Caffe::GetDefaultDevice());
    Blob<float> C(C_shape, Caffe::GetDefaultDevice());
    Blob<float> C_result(C_shape, Caffe::GetDefaultDevice());

    Blob<Dtype> A_quant(A_shape, Caffe::GetDefaultDevice());
    Blob<Dtype> B_quant(B_shape, Caffe::GetDefaultDevice());
    Blob<Dtype> C_quant(C_shape, Caffe::GetDefaultDevice());

    Blob<float> C_unquant(C_shape, Caffe::GetDefaultDevice());


    caffe_rng_gaussian(M * K, (float)-0.25, (float)0.25,
                       A.mutable_cpu_data());
    caffe_rng_gaussian(K * N, (float)-0.25, (float)0.25,
                       B.mutable_cpu_data());
    caffe_rng_gaussian(M * N, (float)-0.25, (float)0.25,
                       C.mutable_cpu_data());

    caffe_copy(M * N, C.cpu_data(), C_result.mutable_cpu_data());

    QuantizerParameter qpm_a;
    QuantizerParameter qpm_b;
    QuantizerParameter qpm_c;
    QuantizerParameter qpm_alpha;
    QuantizerParameter qpm_beta;
    qpm_a.set_mode(CAFFE_QUANT_OBSERVE);
    qpm_b.set_mode(CAFFE_QUANT_OBSERVE);
    qpm_c.set_mode(CAFFE_QUANT_OBSERVE);
    qpm_alpha.set_mode(CAFFE_QUANT_OBSERVE);
    qpm_beta.set_mode(CAFFE_QUANT_OBSERVE);

    Quantizer<float, Dtype> aq(qpm_a);
    Quantizer<float, Dtype> bq(qpm_b);
    Quantizer<float, Dtype> cq(qpm_c);
    Quantizer<float, Dtype> alphaq(qpm_alpha);
    Quantizer<float, Dtype> betaq(qpm_beta);

    // Normal GEMM
    caffe_gemm<float>(
                trans_A, trans_B,
                M, N, K,
                alpha_val,
                A.cpu_data(), B.cpu_data(),
                beta_val,
                C_result.mutable_cpu_data());


    // Observe all values that will be relevant for quantization
    aq.Observe_in_cpu(M * K, A.cpu_data());
    bq.Observe_in_cpu(K * N, B.cpu_data());
    cq.Observe_in_cpu(M * N, C.cpu_data());
    cq.Observe_in_cpu(M * N, C_result.cpu_data());
    alphaq.Observe_in_cpu(1, &alpha_val);
    betaq.Observe_in_cpu(1, &beta_val);

    // Apply observed values to the quantizer
    aq.update();
    bq.update();
    cq.update();
    alphaq.update();
    betaq.update();

    // Quantize A, B and C
    aq.Forward_cpu(M * K, A.cpu_data(), A_quant.mutable_cpu_data());
    bq.Forward_cpu(K * N, B.cpu_data(), B_quant.mutable_cpu_data());
    cq.Forward_cpu(M * N, C.cpu_data(), C_quant.mutable_cpu_data());

    Dtype alpha_val_quant = has_alpha;
    Dtype beta_val_quant = has_beta;

    // Quantize alpha
    if (alpha_with_quant) {
      alphaq.Forward_cpu(1, &alpha_val, &alpha_val_quant);
    }

    // Quantize beta
    if (beta_with_quant) {
      betaq.Forward_cpu(1, &beta_val, &beta_val_quant);
    }

    /*
    std::cout << "C max:" << cq.in_quantizer_values().max << std::endl;
    std::cout << "C min:" << cq.in_quantizer_values().min << std::endl;
    std::cout << "C zero:" << cq.in_quantizer_values().zero << std::endl;
    std::cout << "C scale:" << cq.in_quantizer_values().scale << std::endl;
    std::cout << "C max:" << cq.out_quantizer_values().max << std::endl;
    std::cout << "C min:" << cq.out_quantizer_values().min << std::endl;
    std::cout << "C zero:" << cq.out_quantizer_values().zero << std::endl;
    std::cout << "C scale:" <<  cq.out_quantizer_values().scale << std::endl;
    */

    if (Caffe::mode() == Caffe::Brew::CPU) {
      // Quantized GEMM
      caffe_gemm<Dtype>(
                  trans_A, trans_B,
                  M, N, K,
                  alpha_val_quant,
                  A_quant.cpu_data(), B_quant.cpu_data(),
                  beta_val_quant,
                  C_quant.mutable_cpu_data(),
                  alpha_with_quant ? &(alphaq.out_quantizer_values()) : nullptr,
                  &(aq.out_quantizer_values()),
                  &(bq.out_quantizer_values()),
                  beta_with_quant ? &(betaq.out_quantizer_values()) : nullptr,
                  &(cq.out_quantizer_values()));
    } else {
      Caffe::GetDefaultDevice()->template gemm<Dtype>(trans_A, trans_B,
                  M, N, K,
                  alpha_val_quant,
                  A_quant.gpu_data(), B_quant.gpu_data(),
                  beta_val_quant,
                  C_quant.mutable_gpu_data(),
                  alpha_with_quant ? &(alphaq.out_quantizer_values()) : nullptr,
                  &(aq.out_quantizer_values()),
                  &(bq.out_quantizer_values()),
                  beta_with_quant ? &(betaq.out_quantizer_values()) : nullptr,
                  &(cq.out_quantizer_values()));
    }

    cq.Backward_cpu(M * N, C_quant.cpu_data(), C_unquant.mutable_cpu_data());

    // print_matrix(A_quant.cpu_data(), M, K);
    // print_matrix(B_quant.cpu_data(), K, N);

    // print_matrix(C_quant.cpu_data(), M, N);
    // print_matrix(C_result.cpu_data(), M, N);
    // print_matrix(C_unquant.cpu_data(), M, N);

    const QuantizerValues cqv = cq.in_quantizer_values();
    float eps = std::max(std::abs(cqv.get_max<float>()),
                         std::abs(cqv.get_min<float>()));

    for (int_tp i = 0; i < M * N; ++i) {
      EXPECT_NEAR(C_unquant.cpu_data()[i], C_result.cpu_data()[i], eps);
      // One error is enough to abort
      if (fabs(C_unquant.cpu_data()[i] - C_result.cpu_data()[i]) >= eps) {
        break;
      }
    }
  }
}

}  // namespace caffe
