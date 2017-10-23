// This file is taken from https://github.com/sh1r0/caffe-android-lib
#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <cblas.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#ifdef __cplusplus
extern "C" {
#endif

using string;
using vector;
using caffe::CaffeMobile;

int getTimeSec() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int)now.tv_sec;
}

string jstring2string(JNIEnv *env, jstring jstr) {
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string str(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  return str;
}

/**
 * NOTE: byte[] buf = str.getBytes("US-ASCII")
 */
string bytes2string(JNIEnv *env, jbyteArray buf) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  string s((char *)ptr, env->GetArrayLength(buf));
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return s;
}

cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
  cv::cvtColor(img, img, CV_YUV2RGBA_NV21);
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return img;
}

cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
  return (width == 0 && height == 0) ? cv::imread(bytes2string(env, buf), -1)
                                     : imgbuf2mat(env, buf, width, height);
}

JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setNumThreads(JNIEnv *env,
                                                             jobject thiz,
                                                             jint numThreads) {
  int num_threads = numThreads;
  openblas_set_num_threads(num_threads);
}

JNIEXPORT void JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_enableLog(
    JNIEnv *env, jobject thiz, jboolean enabled) {}

JNIEXPORT jint JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_loadModel(
    JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
  CaffeMobile::Get(jstring2string(env, modelPath),
                   jstring2string(env, weightsPath));
  return 0;
}

JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanFile(
    JNIEnv *env, jobject thiz, jstring meanFile) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetMean(jstring2string(env, meanFile));
}

JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanValues(
    JNIEnv *env, jobject thiz, jfloatArray meanValues) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  int num_channels = env->GetArrayLength(meanValues);
  jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
  vector<float> mean_values(ptr, ptr + num_channels);
  caffe_mobile->SetMean(mean_values);
}

JNIEXPORT void JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setScale(
    JNIEnv *env, jobject thiz, jfloat scale) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetScale(scale);
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_getConfidenceScore(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<float> conf_score =
      caffe_mobile->GetConfidenceScore(getImage(env, buf, width, height));

  jfloatArray result;
  result = env->NewFloatArray(conf_score.size());
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, conf_score.size(), &conf_score[0]);
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jintArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictImage(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height,
    jint k) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<int> top_k =
      caffe_mobile->PredictTopK(getImage(env, buf, width, height), k);

  jintArray result;
  result = env->NewIntArray(k);
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }
  // move from the temp structure to the java structure
  env->SetIntArrayRegion(result, 0, k, &top_k[0]);
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jobjectArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeatures(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height,
    jstring blobNames) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<vector<float>> features = caffe_mobile->ExtractFeatures(
      getImage(env, buf, width, height), jstring2string(env, blobNames));

  jobjectArray array2D =
      env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
  for (size_t i = 0; i < features.size(); ++i) {
    jfloatArray array1D = env->NewFloatArray(features[i].size());
    if (array1D == NULL) {
      return NULL; /* out of memory error thrown */
    }
    // move from the temp structure to the java structure
    env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
    env->SetObjectArrayElement(array2D, i, array1D);
  }
  return array2D;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

//  Compilation fails with error for these lines:
//  FLAGS_redirecttologcat = true;
//  FLAGS_android_logcat_tag = "caffe_jni";

  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
