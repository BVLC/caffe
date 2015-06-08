/** \file convert_imageset_siamese_data.cpp
 * \brief small executable to convert a set of image pairs (files) into a caffe supported format.
 *
 * Converts a set of image pairs (2 filenames on a line in a text file) and the label (last
 * value in the line) into a two channel datum with label provided. This output format is of
 * the same format as the one used in the siamese mnist example.
 *
 * \Notes:
 * Based on code from the Caffe examples and tools
 *
 * \version
 * -	v0.1a	Initial version
 *
 * Future versions:
 * \todo
 * -	add LMDB support
 * -	add RGB channel support -> 6 channels possible?
 * -	Optional encoding of the image so it stored in a smaller format.
 * 		+	It might be not possible without influencing data between the two
 * 			images as they are stored as channels
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

//	Google logging
#include "glog/logging.h"
#include "google/protobuf/text_format.h"

//	Caffe
#include "caffe/common.hpp"
#include "leveldb/db.h"
#include <lmdb.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

//	OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void convert_dataset(const char* input_filename, const char* db_filename, bool signed_data) {

	// Open files
	std::ifstream input_file(input_filename, std::ios::in | std::ios::binary);
	CHECK(input_file) << "Unable to open file " << input_filename;

	// load file contents line by line: [filename] [filename] [label]
	// vector of lines; containing a pair of filenames and a label
	std::vector<std::pair<std::pair<std::string, std::string>, int> > lines;
	std::string filename_1, filename_2;
	int label;
	// read the file
	while (input_file >> filename_1 >> filename_2 >> label) {
		lines.push_back(std::make_pair(std::make_pair(filename_1, filename_2), label));
	}

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
			options, db_filename, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_filename
			<< ". Is it already existing?";
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];

	//	loop over the lines
	caffe::Datum datum;
	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		datum.clear_data();
		datum.clear_float_data();
		datum.set_encoded(false);
		//	ToDo load color images!
		cv::Mat cv_img_1 = cv::imread(lines[line_id].first.first, 0);
		cv::Mat cv_img_2 = cv::imread(lines[line_id].first.second, 0);
		datum.set_channels(2); // one channel for each image in the pair
		datum.set_height(cv_img_1.rows);
		datum.set_width(cv_img_1.cols);
		datum.set_label(lines[line_id].second);

		// if the stored data was originally signed, but converted to unsigned; convert back.
		if(signed_data) {
			cv_img_1.convertTo(cv_img_1, CV_8SC1, 1., -128);
			cv_img_2.convertTo(cv_img_2, CV_8SC1, 1., -128);
		}
		if(	!cv_img_1.empty() && !cv_img_2.empty() &&
			cv_img_1.rows == cv_img_2.rows &&
			cv_img_1.cols == cv_img_2.cols) {
			if(	cv_img_1.depth() == CV_8S &&
				cv_img_2.depth() == CV_8S) {
					std::vector<char> vec_img_1, vec_img_2;
					vec_img_1.assign(cv_img_1.datastart, cv_img_1.dataend);
					vec_img_2.assign(cv_img_2.datastart, cv_img_2.dataend);

					vec_img_1.insert(vec_img_1.end(), vec_img_2.begin(), vec_img_2.end());
					datum.set_data(reinterpret_cast<char*>(&vec_img_1[0]), vec_img_1.size());
			} else
			if(	cv_img_1.depth() == CV_8U &&
				cv_img_2.depth() == CV_8U) {
				std::vector<uchar> vec_img_1, vec_img_2;
				vec_img_1.assign(cv_img_1.datastart, cv_img_1.dataend);
				vec_img_2.assign(cv_img_2.datastart, cv_img_2.dataend);

				vec_img_1.insert(vec_img_1.end(), vec_img_2.begin(), vec_img_2.end());
				datum.set_data(reinterpret_cast<uchar*>(&vec_img_1[0]), vec_img_1.size());
			}
			std::string out;
			datum.SerializeToString(&out);
			snprintf(key, kMaxKeyLength, "%08d", line_id);
			db->Put(leveldb::WriteOptions(), std::string(key), out);
		} else {
			std::cerr 	<< "ERROR!\t one or both input images are empty:\n"
						<< "1: " << lines[line_id].first.first
						<< " is empty: " << cv_img_1.empty() << "\n"
						<< "2: " << lines[line_id].first.second
						<< " is empty: " << cv_img_2.empty() << "\n";
		}
	}

	delete db;
}

int main(int argc, char** argv) {
	switch(argc) {
		case 3:
			google::InitGoogleLogging(argv[0]);
			convert_dataset(argv[1], argv[2], false);
			break;
		case 4:
			google::InitGoogleLogging(argv[0]);
			convert_dataset(argv[1], argv[2], argv[3]);
			break;
		default:
			printf(	"This script converts a image dataset to the leveldb format used\n"
					"by caffe to train a siamese network.\n"
					"Usage:\n"
					"   convert_imageset_siamese_data input_text_file output_db_file signed_data\n"
					"The format of the text file should be:\n"
					"[filename] [filename] [label]\n");
			break;
	}
}

