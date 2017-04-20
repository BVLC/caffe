/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_
#define INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_

#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>
#include "float_compare.hpp"

template <typename DataType>
class Data {
    int dataSize;
    DataType *dataPointer;

    Data(const Data<DataType> &data);
    Data<DataType> &operator =(const Data<DataType> &data);

 public:
        Data() :
            dataSize(0),
            dataPointer(NULL) {
        }

        ~Data() {
            clear();
        }

        int getDataSize() const {
            return dataSize;
        }

        const DataType *getDataPointer() const {
            return dataPointer;
        }

        void clear() {
            delete [] dataPointer;
            dataPointer = NULL;
            dataSize = 0;
        }

        bool loadFromFile(const char *fileName) {
            boost::filesystem::path filePath(fileName);
            if (!boost::filesystem::exists(filePath)) {
                return false;
            }

            if (boost::filesystem::is_empty(filePath)) {
                return false;
            }

            FILE *file = fopen(fileName, "rb");
            if (!file)
                return false;

            int64_t fileSize = boost::filesystem::file_size(filePath);
            DataType *fileDataPointer = new DataType[fileSize];
            size_t bytesRead = fread(fileDataPointer, 1, fileSize, file);
            fclose(file);
            if (bytesRead != fileSize) {
                delete [] fileDataPointer;
                return false;
            }

            clear();

            dataPointer = fileDataPointer;
            dataSize = fileSize / sizeof(DataType);
            return true;
        }
};

class Log {
    FILE *logFile;

    Log() {
        logFile = fopen("log.txt", "w+b");
        CHECK(logFile != NULL) << "Could not open log.txt file";
    }

 public:
        ~Log() {
            if (logFile)
                fclose(logFile);
        }

        static void log(const char *format, ...) {
            va_list args;

            static Log log;

            va_start(args, format);
            vfprintf(log.logFile, format, args);

            va_start(args, format);
            vprintf(format, args);

            va_end(args);
        }
};

void getFileName(char *file_name, bool is_target, const char *name, int id) {
    snprintf(file_name, FILENAME_MAX, "%s%04i.bin", name, id);
}

void getBinFilePath(char *file_path, const char *name) {
    snprintf(file_path, FILENAME_MAX, "%s/%s",
        FLAGS_collect_dir.c_str(), name);
}

bool saveToFile(const char *prefix,
    int id, const float *data, unsigned count) {
    char file_name[FILENAME_MAX];
    getFileName(file_name, false, prefix, id);

    FILE *file = fopen((FLAGS_collect_dir + "/" + file_name).c_str(), "w+b");
    if (!file) {
        LOG(ERROR) << "Failed to create file '" << FLAGS_collect_dir << "'.";
        return false;
    }

    size_t bytesToWrite = count * sizeof(data[0]);
    size_t bytesWritten = fwrite(data, 1, bytesToWrite, file);
    fclose(file);

    if (bytesWritten != bytesToWrite) {
        LOG(ERROR) << "Failed to write data to '" << FLAGS_collect_dir
            << "' file.";
        return false;
    }

    return true;
}

bool loadFromFile(const char *file_path, float *data, unsigned count) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        LOG(ERROR) << "Failed to open file '" << file_path << "' for read.";
        return false;
    }

    size_t bytesToRead = count * sizeof(data[0]);
    size_t bytesRead = fread(data, 1, bytesToRead, file);
    fclose(file);

    if (bytesRead != bytesToRead) {
        LOG(ERROR) << "Failed to read data from '" << file_path << "' file.";
        return false;
    }

    return true;
}

bool compareDataWithFileData(const char *referenceFileName,
  const float *targetDataPointer, double *maxDiff,
  unsigned *diffCounter, const char *outputDir) {
    typedef uint32_t CastType;
    const char *format = "%i;%08X;%08X;%g;%g;%g\n";
    const float epsilon = static_cast<float>(FLAGS_epsilon);
    bool is_nan_filler =
      std::isnan(static_cast<float>(FLAGS_buffer_filler));

    Data<float> referenceData;
    char file_path[FILENAME_MAX];
    getBinFilePath(file_path, referenceFileName);
    if (!referenceData.loadFromFile(file_path)) {
        Log::log("Failed to load reference data file '%s'.\n",
            referenceFileName);
        return false;
    }

    char diffFileName[FILENAME_MAX];
    snprintf(diffFileName, FILENAME_MAX, "./%s/OUT%s", outputDir,
        referenceFileName);
    FILE *file = fopen(diffFileName, "w+t");
    if (!file) {
        return false;
    }

    *maxDiff = -1;
    *diffCounter = 0;
    int dataSize = referenceData.getDataSize();
    const float *referenceDataPointer = referenceData.getDataPointer();
    for (int i = 0; i < dataSize; i++) {
        float a = referenceDataPointer[i];
        float b = targetDataPointer[i];
        if (std::isnan(a) && std::isnan(b) && is_nan_filler){
            continue;
        }

        float diff = caffe::floatDiff(a, b, epsilon);
        if (diff != FP_ZERO) {
            fprintf(file, format, i,(CastType)a, (CastType)b, diff, a, b);
            (*diffCounter)++;
        }

        if (*maxDiff < diff) {
            *maxDiff = diff;
        }

        if (FLAGS_fast_compare && (*diffCounter) >= FLAGS_fast_compare_max) {
            break;
        }
    }

    if (file)
        fclose(file);

    return true;
}

void checkData(const char *referenceFileName, const float *targetDataPointer,
  const char *layerName, const char *outputDir,
  std::unordered_set<string> *erronousLayers) {
    double maxDiff;
    unsigned diffCounter;
    bool success = compareDataWithFileData(referenceFileName,
        targetDataPointer, &maxDiff, &diffCounter, outputDir);

    if (!success) {
        Log::log("%-18s %-20s  : failed\n", referenceFileName, layerName);
    } else if (!diffCounter) {
        Log::log("%-18s %-20s  : success\n", referenceFileName, layerName);
    } else {
        Log::log("%-18s %-20s  : %g %u\n", referenceFileName, layerName,
            maxDiff, diffCounter);
        (*erronousLayers).insert(layerName);
    }
}

void checkAllNans(const float *targetDataPointer, unsigned count,
  const char *bufferName, const char *layerName,
  std::unordered_set<string> *erronousLayers) {
    float buffer_filler = static_cast<float>(FLAGS_buffer_filler);
    float epsilon = static_cast<float>(FLAGS_epsilon);
    if (std::isnan(buffer_filler)){
        for (int i = 0; i < count; i++) {
            if (!std::isnan(targetDataPointer[i])) {
                Log::log("Not all elements in %s are NaNs\n", bufferName);
                (*erronousLayers).insert(layerName);
                return;
            }
        }
    } else {
        for (int i = 0; i < count; i++) {
            if (caffe::floatDiff(targetDataPointer[i], buffer_filler, epsilon)
              != FP_ZERO) {
                Log::log("Not all elements in %s are %.1f\n",
                  bufferName, buffer_filler);
                (*erronousLayers).insert(layerName);
                return;
            }
        }
    }
}

int collectAndCheckLayerData(bool collect_step,
  bool use_gpu, const char *output_dir) {
    Net<float> caffe_net(FLAGS_model, caffe::TRAIN, FLAGS_level,
      NULL, NULL, FLAGS_engine);
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<shared_ptr<Blob<float> > >& params = caffe_net.params();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward =
        caffe_net.bottom_need_backward();

    std::unordered_set<string> erronous_layers;
    FILE *infoFile = fopen(use_gpu ?
        (FLAGS_collect_dir + "/" + "GPUInfo.txt").c_str() :
        (FLAGS_collect_dir + "/" + "CPUInfo.txt").c_str(), "w+t");
    CHECK(infoFile != NULL) << "Could not open info file";
    char file_name[FILENAME_MAX];
    char file_path[FILENAME_MAX];
    string message_prefix = collect_step ? "Collecting" : "Comparing";
    float buffer_filler = static_cast<float>(FLAGS_buffer_filler);
    LOG(INFO) << message_prefix << " weights";
    for (int i = 0; i < params.size(); i++) {
        if (collect_step) {
            saveToFile("Wght", i,
                params[i]->cpu_data(), params[i]->count());
        } else {
            getFileName(file_name, false, "Wght", i);
            checkData(file_name, params[i]->cpu_data(),
                layers[i]->type(), output_dir,
                &erronous_layers);
        }

        caffe::caffe_set(params[i]->count(), buffer_filler,
            params[i]->mutable_cpu_diff());
    }

    LOG(INFO) << message_prefix << " FW Layers";
    for (int i = 0; i < layers.size(); ++i) {
        fprintf(infoFile, "Fwrd%04i %s\n", i, layers[i]->type());

        if (bottom_need_backward[i].size() > 0 && bottom_need_backward[i][0]) {
            if (collect_step) {
                saveToFile("FwrdBtmDat", i, bottom_vecs[i][0]->cpu_data(),
                    bottom_vecs[i][0]->count());
            } else {
                getFileName(file_name, false, "FwrdBtmDat", i);
                getBinFilePath(file_path, file_name);
                loadFromFile(file_path, bottom_vecs[i][0]->mutable_cpu_data(),
                    bottom_vecs[i][0]->count());
            }
        }

        for (int j = 0; j < bottom_vecs[i].size(); j++) {
            caffe::caffe_set(bottom_vecs[i][j]->count(), buffer_filler,
                bottom_vecs[i][j]->mutable_cpu_diff());
        }

        for (int j = 0; j < top_vecs[i].size(); j++) {
            caffe::caffe_set(top_vecs[i][j]->count(), buffer_filler,
                top_vecs[i][j]->mutable_cpu_diff());
        }

        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);

        if (collect_step) {
            saveToFile("FwrdTopDat", i, top_vecs[i][0]->cpu_data(),
                top_vecs[i][0]->count());
        } else {
            getFileName(file_name, false, "FwrdTopDat", i);
            checkData(file_name, top_vecs[i][0]->cpu_data(),
                layers[i]->type(), output_dir,
                &erronous_layers);
        }

        if (bottom_need_backward[i].size() > 0 && bottom_need_backward[i][0]) {
          // We check data only for out-of-place computations
          if (bottom_vecs[i][0] != top_vecs[i][0]) {
              getFileName(file_name, false, "FwrdBtmDat", i);
              checkData(file_name, bottom_vecs[i][0]->cpu_data(),
                  layers[i]->type(), output_dir,
                  &erronous_layers);
          }
          checkAllNans(bottom_vecs[i][0]->cpu_diff(),
              bottom_vecs[i][0]->count(), "bottom.diff",
              layers[i]->type(), &erronous_layers);
        }

        checkAllNans(top_vecs[i][0]->cpu_diff(),
            top_vecs[i][0]->count(), "top.diff",
            layers[i]->type(), &erronous_layers);
    }

    LOG(INFO) << message_prefix
        << " weights again";
    for (int i = 0; i < params.size(); i++) {
        getFileName(file_name, false, "Wght", i);
        checkData(file_name, params[i]->cpu_data(),
            layers[i]->type(), output_dir,
            &erronous_layers);
        checkAllNans(params[i]->cpu_diff(), params[i]->count(), "param.diff",
            layers[i]->type(), &erronous_layers);
    }

    LOG(INFO) << message_prefix << " BW Layers";
    for (int i = layers.size() - 1; i >= 0; --i) {
        fprintf(infoFile, "Bwrd%04i %s\n", i, layers[i]->type());

        layers[i]->Backward(top_vecs[i],
            bottom_need_backward[i], bottom_vecs[i]);

        if (collect_step) {
            saveToFile("BwrdTopDif", i,
                top_vecs[i][0]->cpu_diff(), top_vecs[i][0]->count());

            if (bottom_need_backward[i].size() > 0 &&
                bottom_need_backward[i][0]) {
                saveToFile("BwrdBtmDif", i,
                    bottom_vecs[i][0]->cpu_diff(), bottom_vecs[i][0]->count());
            }
        } else {
            getFileName(file_name, false, "BwrdTopDif", i);
            checkData(file_name, top_vecs[i][0]->cpu_diff(),
                layers[i]->type(), output_dir,
                &erronous_layers);

            if (bottom_need_backward[i].size() > 0 &&
                bottom_need_backward[i][0]) {
                getFileName(file_name, false, "BwrdBtmDif", i);
                checkData(file_name, bottom_vecs[i][0]->cpu_diff(),
                    layers[i]->type(), output_dir,
                    &erronous_layers);
            }
        }
    }

    LOG(INFO) << message_prefix
        << " weights and gradients";
    for (int i = 0; i < params.size(); i++) {
        getFileName(file_name, false, "Wght", i);
        checkData(file_name, params[i]->cpu_data(),
            layers[i]->type(), output_dir,
            &erronous_layers);

        if (collect_step) {
            saveToFile("Grad", i,
                params[i]->cpu_diff(), params[i]->count());
        } else {
            getFileName(file_name, false, "Grad", i);
            checkData(file_name, params[i]->cpu_diff(),
                layers[i]->type(), output_dir,
                &erronous_layers);
        }
    }

    fclose(infoFile);

    if (erronous_layers.size() > 0) {
        LOG(INFO) << "Invalid layer behaviour detected on: ";
        for (const std::string& layer_name : erronous_layers) {
            LOG(WARNING) << "\t" << layer_name;
        }
    }

    return 0;
}

#endif  // INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_
