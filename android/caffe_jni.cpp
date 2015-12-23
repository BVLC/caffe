#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#ifdef __cplusplus
extern "C" {
#endif

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
}

#ifdef __cplusplus
}
#endif
