/** \file convert_mnist_siamese_images.cpp
 * \brief small executable to convert the MNIST database into separate images stored
 * on the filesystem and write pairs in to a file with a similarity label.
 *
 * \Notes:
 * Based on code from the Caffe examples and tools
 *
 * \version
 * -	v0.1a	Initial version
 *
 * Future versions:
 * \todo
 * -	Add image encoding (compression) -> not needed for mnist
 * -	(Re)move the creation of the imageset text file to separate executable
 *
 * \author    	Floris Gaisser <f.gaisser@tudelft.nl>
 * \date      	v0.1a 2015-06-04
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Floris Gaisser, Delft University of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endverbatim
 */

//	C/C++
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

//	Google logging
#include "glog/logging.h"
#include "google/protobuf/text_format.h"

//	Caffe
#include "caffe/common.hpp"
#include "leveldb/db.h"
#include <lmdb.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

//	OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void convert_dataset(const char* image_filename, const char* label_filename, const char* path) {
	// Check dir
	std::string output_path = path;
	if(output_path.find_last_of("/") != output_path.size()-1)
		output_path.append("/");
	struct stat info;
	if(stat( output_path.c_str(), &info ) != 0) {
	    CHECK_EQ(mkdir(output_path.c_str(), 0744), 0)
	        << "mkdir " << output_path << " failed";
	}

    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    CHECK(image_file) << "Unable to open file " << image_filename;
    CHECK(label_file) << "Unable to open file " << label_filename;
    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items, num_labels)
    		<< "number of images (" << num_items << ")"
    		<< "and labels (" << num_labels << ") is not the same.";
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    LOG(INFO) << "A total of " << num_items << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

    //	load the images and save them
    char* pixels = new char[rows * cols];
    char label;
    std::string label_folder_base = output_path + "label_";
    cv::Mat image;
    std::map<char, int> labels;
    std::vector<std::pair<std::string, char> >	image_file_labels;
    for (int itemid = 0; itemid < num_items; ++itemid) {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);

        label += 48;
        std::stringstream ss;
        ss << label_folder_base << std::atoi(&label) << "/";
        // The mnist database is signed, only unsigned values can be stored.
        // ToDo: make it work with signed and unsigned data.
        image = cv::Mat(rows, cols, CV_8SC1, pixels);
        image.convertTo(image, CV_8UC1, 1., 128);

        if(labels.find(label) == labels.end()) {
        	if(stat(ss.str().c_str(), &info ) != 0) {
				CHECK_EQ(mkdir(ss.str().c_str(), 0744), 0)
					<< "mkdir " << ss.str() << " failed";
        	}
    	    labels[label] = 0;
        }
        ss << labels[label] << ".pgm";
        cv::imwrite(ss.str(), image);
        labels[label]++;
        image_file_labels.push_back(std::make_pair(ss.str(), label));
    }

    //	shuffle the image files
    caffe::shuffle(image_file_labels.begin(), image_file_labels.end());

    // create a text file with the file names and the similarity label
	std::ofstream output_file(std::string(output_path + "siamese_set.txt").c_str(), std::ios::out | std::ios::binary);
	CHECK(output_file) << "Unable to open file " << output_file;
    // At least 1 times the total number of images, preferred would be a full cross.
	int scale = 1;
	// round the number of pairs to 100.
	int num_pairs = scale*image_file_labels.size() - ((scale*image_file_labels.size()) % 100);
	std::cout << "making " << num_pairs << " pairs\n";

	// ToDo: change the output amount by param
	// Feature: make a full cross or select only a few closest as similar
    for(int counter = 0; counter < num_pairs; ++counter) {
    	// pick a random  pair
        int i = caffe::caffe_rng_rand() % num_items;
        int j = caffe::caffe_rng_rand() % num_items;
    	// ToDo: check if there are no random double entries

        output_file << image_file_labels.at(i).first << " ";
        output_file << image_file_labels.at(j).first << " ";
        output_file << (image_file_labels.at(i).second == image_file_labels.at(j).second) << "\n";

    }
    output_file.close();

    delete pixels;
}

int main(int argc, char** argv) {
	switch(argc) {
	case 4:
		google::InitGoogleLogging(argv[0]);
		convert_dataset(argv[1], argv[2], argv[3]);
		break;
	default:
		printf(	"This script converts mnist dataset into seperate images and a siamese input textfile\n"
				"This could the be used by create_imageset_siamese.sh\n"
				"Usage:\n"
				"   convert_mnist_siamese_images input_image_file input_label_file output_folder\n");
		break;
	}
	return 0;
}
