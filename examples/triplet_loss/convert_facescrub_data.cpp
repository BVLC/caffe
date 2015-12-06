// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "leveldb/db.h"
#include <leveldb/write_batch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
            "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
            "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
            "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
              "Optional: What type should we encode the image as ('png','jpg',...).");

std::string ReadFileToDatum(const string& filename) {
    std::streampos size;
    fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        std::string buffer(size, ' ');
        file.seekg(0, ios::beg);
        file.read(&buffer[0], size);
        file.close();
        return buffer;
    }
    else {
        return "";
    }
}

std::string CVMatToDatum(const cv::Mat& cv_img) {
    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
    int datum_channels = cv_img.channels();
    int datum_height = cv_img.rows;
    int datum_width = cv_img.cols;
    int datum_size = datum_channels * datum_height * datum_width;
    std::string buffer(datum_size, ' ');
    for (int h = 0; h < datum_height; ++h) {
        const uchar* ptr = cv_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) {
            for (int c = 0; c < datum_channels; ++c) {
                int datum_index = (c * datum_height + h) * datum_width + w;
                buffer[datum_index] = static_cast<char>(ptr[img_index++]);
            }
        }
    }
    return buffer;
}

// Do the file extension and encoding match?
static bool matchExt(const std::string& fn,
                     std::string en) {
    size_t p = fn.rfind('.');
    std::string ext = p != fn.npos ? fn.substr(p) : fn;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    std::transform(en.begin(), en.end(), en.begin(), ::tolower);
    if (ext == en)
        return true;
    if (en == "jpg" && ext == "jpeg")
        return true;
    return false;
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images to the leveldb\n"
                            "format used as input for Caffe.\n"
                            "Usage:\n"
                            "    convert_imageset [FLAGS] ROOTFOLDER/ SIZEFILE LISTFILE DB_NAME \n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 5) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_imageset");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    //const bool encoded = FLAGS_encoded;
    const std::string encode_type = FLAGS_encode_type;

    std::ifstream sizefile(argv[2]);
    std::vector<std::pair<int, int> > num_images_per_identity;
    int identity_id;
    int prefix_sum_num; // inclusive
    while (sizefile >> identity_id >> prefix_sum_num) {
        num_images_per_identity.push_back(std::make_pair(identity_id, prefix_sum_num));
    }

    std::ifstream listfile(argv[3]);
    std::vector<std::pair<int, std::string> > lines;
    std::string filename;
    int face_id;
    while (listfile >> face_id >> identity_id >> filename) {
        lines.push_back(std::make_pair(identity_id, filename));
    }

    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);

    int rows = resize_height;
    int cols = resize_width;

    leveldb::WriteBatch* batch = NULL;

    // Open leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(options, argv[4], &db);
    CHECK(status.ok()) << "Failed to open leveldb " << argv[4]
                       << ". Is it already existing?";
    batch = new leveldb::WriteBatch();

    // Storing to db
    std::string root_folder(argv[1]);
    char* pixels = new char[3 * rows * cols];
    int num_triples = 10000;
    int count = 0;
    std::string value;
    caffe::Datum datum;
    datum.set_channels(3); // one channel for each image in the triple
    datum.set_height(rows);
    datum.set_width(cols);

    //LOG(INFO) << "A total of " << num_items << " items.";
    //LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

    for (int itemid = 0; itemid < num_triples; ++itemid) {
        int anchor_id = caffe::caffe_rng_rand() % num_images_per_identity.size() - 1;                                                         // pick a random  pair
        int anchor = 0;
        int positive = 0;
        while (anchor == positive) {
            anchor = caffe::caffe_rng_rand() % (num_images_per_identity[anchor_id + 1].second - num_images_per_identity[anchor_id].second);   // pick a random  pair
            positive = caffe::caffe_rng_rand() % (num_images_per_identity[anchor_id + 1].second - num_images_per_identity[anchor_id].second); // pick a random  pair
        }
        int negative_id = anchor_id;
        while (anchor_id == negative_id) {
            negative_id = caffe::caffe_rng_rand() % (num_images_per_identity.size() - 1);                                                     // pick a random  pair
        }
        int negative = caffe::caffe_rng_rand() % (num_images_per_identity[negative_id + 1].second - num_images_per_identity[negative_id].second);

        std::string anchor_path = root_folder + "" + lines[num_images_per_identity[anchor_id].second + anchor].second;                        // file to anchor
        std::string positive_path = root_folder + "" + lines[num_images_per_identity[anchor_id].second + positive].second;                    // file to positive
        std::string negative_path = root_folder + "" + lines[num_images_per_identity[negative_id].second + negative].second;                  // file to negative

        std::vector<cv::Mat> cv_img(3);
        cv_img[0] = ReadImageToCVMat(anchor_path, resize_height, resize_width, is_color);
        cv_img[1] = ReadImageToCVMat(positive_path, resize_height, resize_width, is_color);
        cv_img[2] = ReadImageToCVMat(negative_path, resize_height, resize_width, is_color);

        if (cv_img[0].data && cv_img[1].data && cv_img[2].data) {
            if (encode_type.size()) {
                if ((cv_img[0].channels() == 3 && cv_img[1].channels() == 3 && cv_img[2].channels() == 3) == is_color
                    && !resize_height && !resize_width &&
                    matchExt(anchor_path, encode_type) && matchExt(positive_path, encode_type) && matchExt(negative_path, encode_type)) {
                    std::string buffer;
                    buffer.append(ReadFileToDatum(anchor_path));
                    buffer.append(ReadFileToDatum(negative_path));
                    buffer.append(ReadFileToDatum(positive_path));
                    datum.set_data(buffer);
                    datum.set_encoded(true);
                }
                else {
                    std::vector<uchar> buffer;
                    std::vector<uchar> tmp;
                    cv::imencode("." + encode_type, cv_img[0], tmp);
                    buffer.insert(std::end(buffer), std::begin(tmp), std::end(tmp));
                    cv::imencode("." + encode_type, cv_img[1], tmp);
                    buffer.insert(std::end(buffer), std::begin(tmp), std::end(tmp));
                    cv::imencode("." + encode_type, cv_img[2], tmp);
                    buffer.insert(std::end(buffer), std::begin(tmp), std::end(tmp));
                    datum.set_data(std::string(reinterpret_cast<char*>(&buffer[0]), buffer.size()));
                    datum.set_encoded(true);
                }
            }
            else {
                std::string buffer;
                buffer.append(CVMatToDatum(cv_img[0]));
                buffer.append(CVMatToDatum(cv_img[1]));
                buffer.append(CVMatToDatum(cv_img[2]));
                datum.set_data(buffer);
                datum.set_encoded(false);
            }
        }

        datum.SerializeToString(&value);
        std::string key_str = caffe::format_int(itemid, 8);

        batch->Put(key_str, value);

        if (++count % 1000 == 0) {
            db->Write(leveldb::WriteOptions(), batch);
            delete batch;
            batch = new leveldb::WriteBatch();
        }
    }

    // write the last batch
    if (count % 1000 != 0) {
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        delete db;
    }
    delete[] pixels;

#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    return 0;
}
