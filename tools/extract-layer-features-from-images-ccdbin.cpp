#include <cuda_runtime.h>

#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 5) {
    LOG(ERROR) << std::endl
      << "extract-layer-features-from-images" << std::endl
      << "    net-deploy.proto.txt" << std::endl
      << "    pretrained-net-model.proto" << std::endl
      << "    data-mean.proto" << std::endl
      << "    CPU/GPU device-id" << std::endl
      << "    frame-dir file-list" << std::endl
      << "    output-file-base-name layers";
    return 0;
  }

  // Read model paramters
  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  // Read mean file
  LOG(INFO) << "Loading mean file from" << argv[3];
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(argv[3], &blob_proto);
  Blob<float> data_mean_;
  data_mean_.FromProto(blob_proto);
  const float* mean = data_mean_.cpu_data();
  LOG(INFO) << "Load mean done!";

  int device_id = 0;
  Caffe::set_phase(Caffe::TEST);
  if (strcmp(argv[4], "GPU") == 0) {
    device_id = atoi(argv[5]);
    LOG(ERROR) << "Using GPU " << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else if (strcmp(argv[4], "CPU") == 0) {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  // Read file list
  string frame_dir = argv[6];
  std::ifstream infile(argv[7]);
  std::vector<string> frame_names;
  std::vector<string> frame_files;
  string frame_name;
  while(getline(infile, frame_name)) {
    frame_names.push_back(frame_name);
    frame_files.push_back(frame_name);
  }

  // Read layers
  string output_file_base = argv[8];
  int n_layer = argc - 9;
  FILE *output_files[n_layer];
  string layers[n_layer];
  for (int i = 0; i < n_layer; i++) {
    layers[i] = argv[9 + i];
    string output_file_name = output_file_base + ".bin." + argv[9 + i];
    LOG(INFO) << "writing layer " << argv[9 + i] << " into " << output_file_name << std::endl;
    output_files[i] = fopen(output_file_name.c_str(), "wb");
    const int frame_number = frame_files.size();
    const int feat_dim = caffe_test_net.blob_by_name(layers[i])->channels();
    fwrite(&frame_number, sizeof(int), 1, output_files[i]);
    fwrite(&feat_dim, sizeof(int), 1, output_files[i]);
  }

  cv::Mat cv_img, cv_img_orig;
  int n_channel = 3;
  int image_size = 256;
  int image_crop_size = 227;
  int n_image_crop_pixel = n_channel*image_crop_size*image_crop_size;
  int height_offset = 14;
  int width_offset = 14;

  int batch_size = caffe_test_net.blob_by_name("prob")->num();
  std::vector<string> batch_frame_names;
  std::vector<Blob<float>*> batch_frame_blobs;
  Blob<float>* frame_blob = new Blob<float>(batch_size, n_channel, image_crop_size, image_crop_size);

  for (int frame_id = 0; frame_id < frame_files.size(); ++frame_id) {
    batch_frame_names.push_back(frame_names[frame_id]);

    cv_img_orig = cv::imread(frame_dir + "/" + frame_files[frame_id], CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_orig, cv_img, cv::Size(image_size, image_size));

    float* frame_blob_data = frame_blob->mutable_cpu_data();
    for (int i_channel = 0; i_channel < n_channel; ++i_channel) {
      for (int height = 0; height < image_crop_size; ++height) {
        for (int width = 0; width < image_crop_size; ++width) {
          int insertion_index = (frame_id % batch_size)* n_image_crop_pixel + (i_channel * image_crop_size + height) * image_crop_size + width;
          int mean_index = (i_channel * image_size + height + height_offset)*image_size + width + width_offset;
          frame_blob_data[insertion_index] = (static_cast<float>(cv_img.at<cv::Vec3b>(height + height_offset, width + width_offset)[i_channel]) - mean[mean_index]);
        }
      }
    }

    if ((frame_id + 1) % batch_size == 0 || (frame_id + 1) == frame_files.size()) {
      batch_frame_blobs.push_back(frame_blob);
      const vector<Blob<float>*>& result = caffe_test_net.Forward(batch_frame_blobs);

      const int n_example = batch_frame_names.size();
      for (int i_layer = 0; i_layer < n_layer; ++i_layer) {
        const shared_ptr<Blob<float> > data_blob = caffe_test_net.blob_by_name(layers[i_layer]);
        const float* data_blob_ptr = data_blob->cpu_data();
        const int n_channel = data_blob->channels();
	fwrite(data_blob_ptr, sizeof(float), n_example * n_channel, output_files[i_layer]);
      }

      LOG(INFO) << frame_id + 2 << " (+" << n_example
        << ") out of " << frame_files.size() << " results written.";

      batch_frame_names.clear();
      batch_frame_blobs.clear();
    }
  }


  for (int i = 0; i < n_layer; i++) {
    fclose(output_files[i]);
  }

  delete frame_blob;

  return 0;
}
