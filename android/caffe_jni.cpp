#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
<<<<<<< HEAD
#include <vector>
<<<<<<< HEAD

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif
=======
>>>>>>> 4b51010... Revised jni interfaces

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif
=======
>>>>>>> 25d8ecc... Added jni lib for android

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

<<<<<<< HEAD
=======
#define  LOG_TAG    "caffe-mobile"
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG, __VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)

>>>>>>> 25d8ecc... Added jni lib for android
#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD
using std::string;
using std::vector;
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

JNIEXPORT void JNICALL
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7803d98... Add setNumThreads for Eigen/OpenBLAS
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setNumThreads(
    JNIEnv *env, jobject thiz, jint numThreads) {
  int num_threads = numThreads;
#ifdef USE_EIGEN
  omp_set_num_threads(num_threads);
#else
  openblas_set_num_threads(num_threads);
#endif
}

JNIEXPORT void JNICALL
<<<<<<< HEAD
=======
>>>>>>> 4b51010... Revised jni interfaces
=======
>>>>>>> 7803d98... Add setNumThreads for Eigen/OpenBLAS
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_enableLog(JNIEnv *env,
                                                         jobject thiz,
                                                         jboolean enabled) {}

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

JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setScale(JNIEnv *env,
                                                        jobject thiz,
                                                        jfloat scale) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  caffe_mobile->SetScale(scale);
}

JNIEXPORT jintArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictImage(JNIEnv *env,
                                                            jobject thiz,
                                                            jstring imgPath,
                                                            jint k) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<int> top_k =
      caffe_mobile->PredictTopK(jstring2string(env, imgPath), k);

  jintArray result;
  result = env->NewIntArray(k);
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }
  // move from the temp structure to the java structure
  env->SetIntArrayRegion(result, 0, k, &top_k[0]);
  return result;
}

JNIEXPORT jobjectArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeatures(
    JNIEnv *env, jobject thiz, jstring imgPath, jstring blobNames) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get();
  vector<vector<float>> features = caffe_mobile->ExtractFeatures(
      jstring2string(env, imgPath), jstring2string(env, blobNames));

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

  return JNI_VERSION_1_6;
=======
caffe::CaffeMobile *caffe_mobile;

int getTimeSec();

__NDK_FPABI__ void JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_CaffeMobile_enableLog(JNIEnv* env, jobject thiz, jboolean enabled)
{
}

__NDK_FPABI__ jint JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_CaffeMobile_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jstring weightsPath)
{
    const char *model_path = env->GetStringUTFChars(modelPath, 0);
    const char *weights_path = env->GetStringUTFChars(weightsPath, 0);
    caffe_mobile = new caffe::CaffeMobile(string(model_path), string(weights_path));
    env->ReleaseStringUTFChars(modelPath, model_path);
    env->ReleaseStringUTFChars(weightsPath, weights_path);
    return 0;
}

__NDK_FPABI__ jint JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_CaffeMobile_predictImage(JNIEnv* env, jobject thiz, jstring imgPath)
{
    const char *img_path = env->GetStringUTFChars(imgPath, 0);
    caffe::vector<int> top_k = caffe_mobile->predict_top_k(string(img_path), 3);
    LOGD("top-1 result: %d", top_k[0]);

    env->ReleaseStringUTFChars(imgPath, img_path);

    return top_k[0];
}

int getTimeSec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int) now.tv_sec;
}
/*
JavaVM *g_jvm = NULL;
jobject g_obj = NULL;

void JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_MainActivity_MainActivity_setJNIEnv(JNIEnv* env, jobject obj)
{
    env->GetJavaVM(&g_jvm);
    g_obj = env->NewGlobalRef(obj);
}
*/
jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    JNIEnv* env = NULL;
    jint result = -1;

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        LOGE("GetEnv failed!");
        return result;
    }

    return JNI_VERSION_1_6;
>>>>>>> 25d8ecc... Added jni lib for android
}

#ifdef __cplusplus
}
#endif
<<<<<<< HEAD
=======

int main(int argc, char const *argv[])
{
    string usage("usage: main <model> <weights> <img>");
    if (argc < 4) {
        std::cerr << usage << std::endl;
        return 1;
    }

    caffe_mobile = new caffe::CaffeMobile(string(argv[1]), string(argv[2]));
    caffe::vector<int> top_3 = caffe_mobile->predict_top_k(string(argv[3]));
    for (auto k : top_3) {
        std::cout << k << std::endl;
    }
    return 0;
}

>>>>>>> 25d8ecc... Added jni lib for android
