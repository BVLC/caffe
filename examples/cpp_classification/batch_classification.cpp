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

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gflags/gflags.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

DEFINE_string(model, "",
    "Required; The model definition protocol buffer text file.");

DEFINE_string(weights, "",
    "Required; The pretrained weights.");

DEFINE_string(input, "",
    "Required; File that contain the path of input images line by line");

DEFINE_string(label_file, "",
    "Required; The label file.");

DEFINE_string(engine, "",
    "Optional; Engine can only be CAFFE | MKL2017 | MKLDNN");

DEFINE_string(mean_file, "",
    "Optional; The mean file used to subtract from the input image.");

DEFINE_string(mean_value, "104,117,123",
    "Optional; If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','.");

DEFINE_int32(batch_size, 1,
    "Optional; batch size, default 1");

typedef std::pair<string, float> Prediction;

class Classifier {
    public:
        Classifier(const string& model_file,
                const string& trained_file,
                const string& mean_file,
                const string& mean_value,
                const string& label_file,
                const string& engine,
                const size_t batch_size,
                const size_t topN = 5
                );
        vector<vector<Prediction> > ClassifyBatch(vector<cv::Mat>& imgs);

    private:
        void SetMean(const string& mean_file, const string& mean_value);

        vector<float> PredictBatch(vector<cv::Mat>& imgs);

        void WrapInputLayerBatch(vector<vector<cv::Mat> >* input_channels_batch);
        void WriteImgToInput(const vector<cv::Mat>& imgs, vector<vector<cv::Mat> >* input_channels_batch);
        void Preprocess(cv::Mat& img);

        void PreprocessBatch(vector<cv::Mat>& imgs);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        size_t batch_size_;
        size_t topN_;
        std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
        const string& trained_file,
        const string& mean_file,
        const string& mean_value,
        const string& label_file,
        const string& engine,
        const size_t batch_size,
        const size_t topN
        ) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST, 0, NULL, NULL, engine));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    SetMean(mean_file, mean_value);

    batch_size_ = batch_size;
    topN_ = topN;

    if(!label_file.empty()) {
    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
    }

}


static bool PairCompare(const std::pair<float, int>& lhs,
        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static vector<int> Argmax(const vector<float>& v, int N) {
    vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
vector<vector<Prediction> > Classifier::ClassifyBatch(vector<cv::Mat>& imgs) {
    vector<float> output_batch = PredictBatch(imgs);
    vector<vector<Prediction> > predictionsBatch;
    int output_channels = net_->output_blobs()[0]->channels();
    for (size_t i = 0; i < batch_size_; ++i) {
        vector<float> output(output_batch.begin() + i*output_channels, output_batch.begin()+(i+1)*output_channels);
        vector<int> maxN = Argmax(output, topN_);
        vector<Prediction>  predictions;
        for (int i = 0; i < topN_; ++i) {
            int idx = maxN[i];
            if(labels_.empty()) {
                predictions.push_back(std::make_pair(std::to_string(idx), output[idx]));
            } else{
                predictions.push_back(std::make_pair(labels_[idx], output[idx]));
            }
        }
        predictionsBatch.push_back(predictions);
    }
    return predictionsBatch;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if(!mean_file.empty()) {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
            "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

vector<float> Classifier::PredictBatch(vector<cv::Mat>& imgs) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(batch_size_, num_channels_,
            input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<vector<cv::Mat> > input_channels_batch;
    WrapInputLayerBatch(&input_channels_batch);
    PreprocessBatch(imgs);
    WriteImgToInput(imgs, &input_channels_batch);

    net_->Forward();

    /* Copy the output layer to a vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels() * batch_size_;
    printf("output_layer->channels: %d\n", output_layer->channels());
    return vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayerBatch(vector<vector<cv::Mat> >* input_channels_batch) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    int num = input_layer->num();
    for( int j = 0; j < num; ++j) {
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_channels_batch->push_back(input_channels);
    }
}

void Classifier::WriteImgToInput(const vector<cv::Mat>& imgs,
        vector<vector<cv::Mat> >* input_channels_batch)
{
    for(size_t i=0; i<batch_size_; ++i) {
        cv::split(imgs[i], input_channels_batch->at(i));
    }
}

void Classifier::PreprocessBatch(vector<cv::Mat>& imgs) {
    for(size_t i=0; i<imgs.size(); ++i) {
        Preprocess(imgs[i]);
    }
}

void Classifier::Preprocess(cv::Mat& img) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

//    cv::Mat sample_normalized;
//    cv::subtract(sample_float, mean_, sample_normalized);
    cv::subtract(sample_float, mean_, img);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
//    cv::split(sample_normalized, *input_channels);

}

vector<cv::Mat> loadImgBatch(vector<string> imgNames) {
    vector<cv::Mat> imgs;
    for(size_t i=0; i<imgNames.size(); ++i) {
        cv::Mat img = cv::imread(imgNames[i], -1);
        CHECK(!img.empty()) << "Unable to decode image " << imgNames[i];
        imgs.push_back(img);
    }
    return imgs;
}

void printPrediction(vector<Prediction> predictions) {
    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        cout << std::fixed << std::setprecision(4) << p.second << " - \""
            << p.first << "\"" << endl;
    }
}

void printPredictionsBatch(vector<string> imgNames,
        vector<vector<Prediction> > predictionsBatch) {
    for( size_t i = 0; i < predictionsBatch.size(); ++i) {
        cout << "---------- "<< i + 1 <<": Prediction for "
            << imgNames[i] << " ----------" << endl;
        printPrediction(predictionsBatch[i]);
    }
}

vector<string> readImgListFromPath(string file) {
    vector<string> rawImgNames;
    std::ifstream input_lines(file.c_str());
    CHECK(input_lines) << "Unable to open file " << file;
    string line;
    while (std::getline(input_lines, line))
        rawImgNames.push_back(string(line));
    return rawImgNames;
}
int main(int argc, char** argv) {

    ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Image classification.\n"
        "Usage:\n"
        "batch_classification <args>\n"
        "Example: ./batch_classification --model <model path> --weights <weights path> --input <input.txt> --batch_size <num>"
        );
    gflags::ParseCommandLineFlags(&argc, &argv, true);


    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_input.size(), 0) << "Need model weights to score.";

    cout<<"Use batch size: "<< FLAGS_batch_size << endl;

    if (FLAGS_mean_file.empty()) {
        cout<<"Use mean value: "<< FLAGS_mean_value<<endl;
    }else{
        cout<<"Use mean file: "<<FLAGS_mean_file<<endl;
    }

    Classifier classifier(FLAGS_model, FLAGS_weights, FLAGS_mean_file,
            FLAGS_mean_value, FLAGS_label_file, FLAGS_engine, FLAGS_batch_size);

    vector<string> rawImgNames = readImgListFromPath(FLAGS_input);

    if(rawImgNames.size() > 0 && rawImgNames.size() < FLAGS_batch_size) {
        while(rawImgNames.size() < FLAGS_batch_size) {
            rawImgNames.insert(rawImgNames.end(), rawImgNames.begin(), rawImgNames.end());
        }
    }

    vector<string> imgNames(rawImgNames.begin(), rawImgNames.begin() + FLAGS_batch_size);
    vector<cv::Mat> imgs = loadImgBatch(rawImgNames);

    vector<vector<Prediction> > predictionsBatch = classifier.ClassifyBatch(imgs);

    printPredictionsBatch(imgNames, predictionsBatch);

    return 0;
}



#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
