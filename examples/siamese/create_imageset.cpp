/** \file create_imageset.cpp
 * \brief small executable to create a file with image pairs with a similarity
 * label.
 *
 * Loads the paths to a labeled set of images; Each folder is a label and all
 * the images contained in that folder belongs to that label.
 * A set of image pairs and their similarity label are selected. This can be
 * done with the following criteria:
 * 	-	random selection of images with a fixed set size.
 * 		+	check for doubles
 * 	-	full cross of all similar and dissimilar images.
 * 		+	limit the number of similar pairs
 * 		+	limit the number of dissimilar pairs
 * 	-	Limited cross of similar and dissimilar images using Euclidean
 * 	    difference.
 * 		+	limit the number of similar pairs
 * 		+	limit the number of dissimilar pairs
 *
 * \Notes:
 * Based on code from the Caffe examples and tools
 *
 * \version
 * -	v0.1a	Initial version:
 * 				Random selection of similar and dissimilar images
 *
 * Future versions:
 * -	v0.1b	Full and limited cross of (dis)similar images
 * -	v0.1c	Limited cross of (dis)similar images using Euclidean difference
 *
 * \author    	Floris Gaisser <f.gaisser@tudelft.nl>
 * \date      	v0.1a 2015-06-08
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
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <vector>

//	Google logging
#include "glog/logging.h"
#include "google/protobuf/text_format.h"

//	Caffe
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "leveldb/db.h"
#include <lmdb.h>

//	OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


/*
 * load pairs of image paths and their label
 * ToDo: Move to some more general location
 */
void loadImagePaths(
        std::string input_folder,
        std::string file_extention,
        std::vector<std::pair<std::string, std::string> > &images) {

    //  check for trailing '/'
    if(input_folder.find_last_of("/") != input_folder.size()-1)
        input_folder.append("/");
    if(!file_extention.empty() && file_extention.find(".") != 0) {
        file_extention = "." + file_extention;
    }

    //  check dir
    DIR *input_dir_handle, *label_dir_handle;
    struct dirent *input_dir_contents_handle, *label_dir_contents_handle;
    if((input_dir_handle = opendir(input_folder.c_str())) == NULL) {
        std::cerr << "Folder " << input_folder << " does not exist!\n";
    } else { // dir exists
        while((input_dir_contents_handle = readdir(input_dir_handle)) != NULL) {
            //  discard files and folders starting with '.'
            if(std::string(input_dir_contents_handle->d_name).at(0) == '.') {
                continue;
            }
            if((label_dir_handle = opendir(std::string(
                    input_folder + std::string(input_dir_contents_handle->d_name)).c_str()))
                != NULL) {
                //  this is a folder -> thus a label
                while((label_dir_contents_handle = readdir(label_dir_handle)) != NULL) {
                    //  discard files and folders starting with '.'
                    if(std::string(input_dir_contents_handle->d_name).at(0) == '.') {
                        continue;
                    }
                    if( !file_extention.empty() &&
                        std::string(label_dir_contents_handle->d_name).find(file_extention.c_str())
                            == std::string::npos) {
                        continue;
                    }
                    std::string filename = input_folder +
                            std::string(input_dir_contents_handle->d_name) + "/" +
                            std::string(label_dir_contents_handle->d_name);
                    images.push_back(   std::make_pair(
                                            filename,
                                            input_dir_contents_handle->d_name));
                }
            }
        }
    }
}

/*
 * load image paths grouped by label
 * ToDo: Move to some more general location
 */
int loadImagePathsByLabel(
        std::string input_folder,
        std::string file_extention,
        std::map<std::string, std::vector<std::string> > &images) {

    int total_images = 0;
    //  check for trailing '/'
    if(input_folder.find_last_of("/") != input_folder.size()-1)
        input_folder.append("/");
    if(!file_extention.empty() && file_extention.find(".") != 0) {
        file_extention = "." + file_extention;
    }

    //  check dir
    DIR *input_dir_handle, *label_dir_handle;
    struct dirent *input_dir_contents_handle, *label_dir_contents_handle;
    if((input_dir_handle = opendir(input_folder.c_str())) == NULL) {
        std::cerr << "Folder " << input_folder << " does not exist!\n";
    } else { // dir exists
        while((input_dir_contents_handle = readdir(input_dir_handle)) != NULL) {
            //  discard files and folders starting with '.'
            if(std::string(input_dir_contents_handle->d_name).at(0) == '.') {
                continue;
            }
            if((label_dir_handle = opendir(std::string(
                    input_folder + std::string(input_dir_contents_handle->d_name)).c_str()))
                != NULL) {
                //  this is a folder -> thus a label
                images[input_dir_contents_handle->d_name] = std::vector<std::string>();
                while((label_dir_contents_handle = readdir(label_dir_handle)) != NULL) {
                    //  discard files and folders starting with '.'
                    if(std::string(label_dir_contents_handle->d_name).at(0) == '.') {
                        continue;
                    }
                    if( !file_extention.empty() &&
                        std::string(label_dir_contents_handle->d_name).find(file_extention.c_str())
                            == std::string::npos) {
                        continue;
                    }
                    std::string filename = input_folder +
                            std::string(input_dir_contents_handle->d_name) + "/" +
                            std::string(label_dir_contents_handle->d_name);
                    images[input_dir_contents_handle->d_name].push_back(filename);
                    total_images++;
                }
            }
        }
    }
    return total_images;
}

/*
 * load image images grouped by label
 * ToDo: Move to some more general location
 */
int loadImagesByLabel(
        std::string input_folder,
        std::string file_extention,
        std::map<std::string, std::vector<cv::Mat> > &images) {

    int total_images = 0;
    //  check for trailing '/'
    if(input_folder.find_last_of("/") != input_folder.size()-1)
        input_folder.append("/");
    if(!file_extention.empty() && file_extention.find(".") != 0) {
        file_extention = "." + file_extention;
    }

    //  check dir
    DIR *input_dir_handle, *label_dir_handle;
    struct dirent *input_dir_contents_handle, *label_dir_contents_handle;
    if((input_dir_handle = opendir(input_folder.c_str())) == NULL) {
        std::cerr << "Folder " << input_folder << " does not exist!\n";
    } else { // dir exists
        while((input_dir_contents_handle = readdir(input_dir_handle)) != NULL) {
            //  discard files and folders starting with '.'
            if(std::string(input_dir_contents_handle->d_name).at(0) == '.') {
                continue;
            }
            if((label_dir_handle = opendir(std::string(
                    input_folder + std::string(input_dir_contents_handle->d_name)).c_str()))
                != NULL) {
                //  this is a folder -> thus a label
                images[input_dir_contents_handle->d_name] = std::vector<cv::Mat>();
                while((label_dir_contents_handle = readdir(label_dir_handle)) != NULL) {
                    //  discard files and folders starting with '.'
                    if(std::string(label_dir_contents_handle->d_name).at(0) == '.') {
                        continue;
                    }
                    if(!file_extention.empty() &&
                        std::string(label_dir_contents_handle->d_name).find(file_extention.c_str())
                            == std::string::npos) {
                        continue;
                    }
                    std::string filename = input_folder +
                            std::string(input_dir_contents_handle->d_name) + "/" +
                            std::string(label_dir_contents_handle->d_name);
                    images[input_dir_contents_handle->d_name].push_back(cv::imread(filename));
                    if(images[input_dir_contents_handle->d_name].back().empty()) {
                        std::cerr << "help! empty image: " << filename << "\n";
                    }
                    total_images++;
                }
            }
        }
    }
    return total_images;
}

bool checkAddForPair(
        std::string i,
        std::string j,
        std::vector<std::string> &pair_hashes) {
    std::string ss, ss_reverse;
    ss = i + "_" + j;
    ss_reverse = j + "_" + i;
    if(std::find(pair_hashes.begin(), pair_hashes.end(), ss) == pair_hashes.end()) {
        // not found, so add
        pair_hashes.push_back(ss);
        pair_hashes.push_back(ss_reverse);
        return false;
    }
    return true;
}
bool checkAddForLabeledPair(
        int i,
        std::string label_i,
        int j,
        std::string label_j,
        std::vector<std::string> &pair_hashes) {
    std::stringstream ss_i, ss_j;
    ss_i << label_i << "-" << i;
    ss_j << label_j << "-" << j;
    return checkAddForPair(ss_i.str(), ss_j.str(), pair_hashes);
}
bool checkAddForPair(
        int i,
        int j,
        std::vector<std::string> &pair_hashes) {
    std::stringstream ss_i, ss_j;
    ss_i << i;
    ss_j << j;
    return checkAddForPair(ss_i.str(), ss_j.str(), pair_hashes);
}

/*
 * Select random pairs and write to file
 *
 */
void selectRandom(
        std::vector<std::pair<std::string, std::string> > &images,
        std::string output_filename,
        int limit) {
    //  create the output file
    std::ofstream output_file(output_filename.c_str(), std::ios::out | std::ios::binary);
    CHECK(output_file) << "Unable to open file " << output_filename;

    // shuffle the image files
    caffe::shuffle(images.begin(), images.end());

    std::vector<std::string> pair_hashes;
    int num_images = images.size();
    for(int counter = 0; counter < limit; ) {
        // pick a random pair
        int i = caffe::caffe_rng_rand() % num_images;
        int j = caffe::caffe_rng_rand() % num_images;

        // check if there are no random double entries
        if(!checkAddForPair(i, j, pair_hashes)) {
            output_file << images.at(i).first << " ";
            output_file << images.at(j).first << " ";
            output_file << (images.at(i).second == images.at(j).second) << "\n";
        }
//        std::stringstream ss, ss_reverse;
//        ss << i << "_" << j;
//        ss_reverse << j << "_" << i;
//        if(std::find(pair_hashes.begin(), pair_hashes.end(), ss.str()) == pair_hashes.end()) {
//            output_file << images.at(i).first << " ";
//            output_file << images.at(j).first << " ";
//            output_file << (images.at(i).second == images.at(j).second) << "\n";
//            pair_hashes.push_back(ss.str());
//            pair_hashes.push_back(ss_reverse.str());
//            counter++;
//        }
    }
    output_file.close();
}

void selectCross(
        std::map<std::string, std::vector<std::string> > &imagePaths,
        std::string output_filename,
        int similar_limit = 100,
        int dissimilar_limit = 1000) {
    bool full_similar = false;
    bool full_dissimilar = false;
    if(similar_limit <= 0) {
        full_similar = true;
    }
    if(dissimilar_limit <= 0) {
        full_dissimilar = true;
    }

    //  create the output file
    std::ofstream output_file(output_filename.c_str(), std::ios::out | std::ios::binary);
    CHECK(output_file) << "Unable to open file " << output_filename;

    std::vector<std::pair<std::pair<std::string, std::string>, int> > selected_image_pairs;
    std::map<std::string, std::vector<std::string> >::iterator label_it, other_label_it;
    //  loop over the labels
    for(label_it = imagePaths.begin(); label_it != imagePaths.end(); label_it++) {
        //  loop over the images
        std::cout   << "Start processing label: " << label_it->first << ".\n";
        for(int image_counter = 0; image_counter < label_it->second.size(); image_counter++) {
            // select similar
            if(full_similar) {
                for(int similar_counter = 0; similar_counter < label_it->second.size(); similar_counter++) {
                    if(similar_counter == image_counter) { continue; } // don't pair to yourself!
                    selected_image_pairs.push_back(
                            std::make_pair(
                                    std::make_pair(
                                            imagePaths[label_it->first].at(image_counter),
                                            imagePaths[label_it->first].at(similar_counter)),
                                    1));
                }
            } else {
                std::vector<std::string> pair_hashes;
                int num_images = label_it->second.size();
                // pick a random pair
                for(int similar_counter = 0; similar_counter < similar_limit; ) {
                    int i = caffe::caffe_rng_rand() % num_images;
                    int j = caffe::caffe_rng_rand() % num_images;
                    if(i == j) { continue; } // don't pair to yourself!

                    // check if there are no random double entries
                    if(!checkAddForPair(i, j, pair_hashes)) {
                        selected_image_pairs.push_back(
                                std::make_pair(
                                        std::make_pair(
                                                imagePaths[label_it->first].at(i),
                                                imagePaths[label_it->first].at(j)),
                                        1));
                        similar_counter++;
                    }
                }
            }

            //  select dissimilar
            if(full_dissimilar) {
                for(other_label_it = label_it; other_label_it != imagePaths.end(); other_label_it++) {
                    if(label_it->first == other_label_it->first) { continue; } // skip if the labels are the same!
                    for(int dissimilar_counter = 0; dissimilar_counter < other_label_it->second.size(); dissimilar_counter++) {
                        selected_image_pairs.push_back(
                                std::make_pair(
                                        std::make_pair(
                                                imagePaths[label_it->first].at(image_counter),
                                                imagePaths[other_label_it->first].at(dissimilar_counter)),
                                        0));
                    }
                }
            } else {
                std::vector<std::string> pair_hashes;
                int num_labels = imagePaths.size();
                // pick a random pair
                for(int dissimilar_counter = 0; dissimilar_counter < dissimilar_limit; ) {
                    int label_j = caffe::caffe_rng_rand() % num_labels;
                    other_label_it = imagePaths.begin();
                    std::advance(other_label_it, label_j);
                    if(other_label_it == label_it) { continue; } // don't pair to same label
                    if(other_label_it == imagePaths.end()) { continue; } // check for out of bounds
                    int i = caffe::caffe_rng_rand() % label_it->second.size();
                    int j = caffe::caffe_rng_rand() % other_label_it->second.size();

                    // check if there are no random double entries
//                    std::cout   << label_it->first << " @ "
//                                << i << " < "
//                                << label_it->second.size() << "\t\t"
//                                << other_label_it->first << " @ "
//                                << j << " < "
//                                << other_label_it->second.size() << "\n";
                    if(!checkAddForLabeledPair(i, label_it->first, j, other_label_it->first, pair_hashes)) {
                        selected_image_pairs.push_back(
                                std::make_pair(
                                        std::make_pair(
                                                imagePaths[label_it->first].at(i),
                                                imagePaths[other_label_it->first].at(j)),
                                        0));
                        dissimilar_counter++;
                    }
                }
            }
            if(image_counter % 100 == 0) {
                std::cout   << "processed: " << image_counter << " images in label: " << label_it->first << ".\n";
            }
        }
        std::cout   << "label: " << label_it->first << " processed.\n"
                    << "Created: " << selected_image_pairs.size() << " up to now!\n";
    }

    // shuffle the image files
    std::cout << "Shuffling images...\n";
    caffe::shuffle(selected_image_pairs.begin(), selected_image_pairs.end());

    std::cout << "Writing " << selected_image_pairs.size() << " pairs to file...\n";
    std::vector<std::pair<std::pair<std::string, std::string>, int> >::iterator pair_it;
    for(pair_it = selected_image_pairs.begin(); pair_it != selected_image_pairs.end(); pair_it++) {
        output_file << pair_it->first.first << " " << pair_it->first.second << " " << pair_it->second << "\n";
    }
    output_file.close();
}


bool compare_pairs(std::pair<std::string, double> a, std::pair<std::string, double> b) {
    return (a.second < b.second);
}
/*
 * Select the closest (dis)similar images based on L1 norm distance.
 *
 * ToDo: Clean up code into small sub-functions
 */
void selectClosest(
        std::map<std::string, std::vector<cv::Mat> > &images,
        std::map<std::string, std::vector<std::string> > &imagePaths,
        std::string output_filename,
        int total_image,
        int similar_limit = 5,
        int dissimilar_limit = 50) {
    if(similar_limit <= 0) {
        similar_limit = 5;
    }
    if(dissimilar_limit <= 0) {
        dissimilar_limit = 50;
    }

    std::vector<std::pair<std::pair<std::string, std::string>, int> > selected_image_pairs;
    std::map<std::string, std::vector<cv::Mat> >::iterator label_it, other_label_it;
    //  create the output file
    std::ofstream output_file(output_filename.c_str(), std::ios::out | std::ios::binary);
    CHECK(output_file) << "Unable to open file " << output_filename;
    cv::Mat curr_image, match_image;
    //  loop over the labels
    for(label_it = images.begin(); label_it != images.end(); label_it++) {
        //  loop over the images
        std::cout   << "Start processing label: " << label_it->first << ".\n";
        for(int image_counter = 0; image_counter < label_it->second.size(); image_counter++) {
            //  find closest similar images
            curr_image = label_it->second.at(image_counter);
            std::vector<std::pair<std::string, double> > closest;
            for(int similar_counter = 0; similar_counter < label_it->second.size(); similar_counter++) {
                if(similar_counter == image_counter) { continue; }  // don't match to yourself
                match_image = label_it->second.at(similar_counter);

                double distance = cv::norm(curr_image, match_image, cv::NORM_L1);

                closest.push_back(std::make_pair(imagePaths[label_it->first].at(similar_counter), distance));
                std::sort(closest.begin(), closest.end(), compare_pairs);
                if(closest.size() > similar_limit) {
                    closest.pop_back();
                }
            }
            //  looped over all similar images, found most similar images:
            for(int match_counter = 0; match_counter < closest.size(); match_counter++) {
                selected_image_pairs.push_back(
                        std::make_pair(
                                std::make_pair(
                                        imagePaths[label_it->first].at(image_counter),
                                        closest.at(match_counter).first),
                                1));
            }

            //  find closest dissimilar images
            closest.clear();
            for(other_label_it = images.begin(); other_label_it != images.end(); other_label_it++) {
                if(label_it->first == other_label_it->first) { continue; } // skip if the labels are the same!
                for(int dissimilar_counter = 0; dissimilar_counter < other_label_it->second.size(); dissimilar_counter++) {
                    match_image = other_label_it->second.at(dissimilar_counter);
                    double distance = cv::norm(curr_image, match_image, cv::NORM_L1);

                    closest.push_back(std::make_pair(
                            imagePaths[other_label_it->first].at(dissimilar_counter),
                            distance));

                    std::sort(closest.begin(), closest.end(), compare_pairs);
                    if(closest.size() > dissimilar_limit) {
                        closest.pop_back();
                    }
                }
            }
            //  looped over all dissimilar images, found most dissimilar images:
            for(int match_counter = 0; match_counter < closest.size(); match_counter++) {
                selected_image_pairs.push_back(
                        std::make_pair(
                                std::make_pair(
                                        imagePaths[label_it->first].at(image_counter),
                                        closest.at(match_counter).first),
                                0));
            }
            if(image_counter % 100 == 0) {
                std::cout   << "processed: " << image_counter << " images in label: " << label_it->first << ".\n";
            }
        }
        std::cout   << "label: " << label_it->first << " processed.\n"
                    << "Created: " << selected_image_pairs.size() << " up to now!\n";
    }

    // shuffle the image files
    std::cout << "Shuffling images...\n";
    caffe::shuffle(selected_image_pairs.begin(), selected_image_pairs.end());

    std::cout << "Writing " << selected_image_pairs.size() << " pairs to file...\n";
    std::vector<std::pair<std::pair<std::string, std::string>, int> >::iterator pair_it;
    for(pair_it = selected_image_pairs.begin(); pair_it != selected_image_pairs.end(); pair_it++) {
        output_file << pair_it->first.first << " " << pair_it->first.second << " " << pair_it->second << "\n";
    }
    output_file.close();
}

int main(int argc, char** argv) {
	//	0.	vars
	bool			random_select = false;
	bool			full_cross = false;
	bool			closest_cross = false;
	int				limit = 0;
	int				similar_limit = 0;
	int				dissimilar_limit = 0;
	std::string		output_filename = "";
	std::string 	input_folder = "";
	std::string 	file_extention = "";

    //  1.  Get arguments
    for(int i = 1; i < argc; i++) {
        if(argv[i][0] == '-') { // we have option
            int i_counter = 0;
            for(unsigned int j = 1; j < strlen(argv[i]); j++) {
                char param = argv[i][j];
                //std::cout << "option: " << param << "\n";
                switch(param) {
                    case 'h': // help
                    //case '-help': // help
                        std::cout   << "-o [file]\t Output file\n"
                                    << "-i [folder]\t input folder containing labeled folders with images\n"
                                    << "-e [ext]\t load images with file extention\n"
                                    << "-r (int)\t Random selection; optional size\n"
                                    << "-f (int) (int)\t full or limited (dis)similar cross\n"
                                    << "-c [int] [int]\t closest limited cross\n"
                                    << "Note: set the limit to match batch sizes.";
                        return 0;
                        break;
                    case 'o':   //  output file
                        i_counter++;
                        output_filename = std::string(argv[i+i_counter]);
                        break;
                    case 'i':   //  input folder
                        i_counter++;
                        input_folder = std::string(argv[i+i_counter]);
                        break;
                    case 'r':   //  random selection
                        random_select = true;
                        i_counter++;
                        if(argc > (i+i_counter) && argv[i+i_counter][0] != '-') {
                            limit = std::atoi(argv[i+i_counter]);
                        }
                        break;
                    case 'f':   //  full cross
                        full_cross = true;
                        i_counter++;
                        if(argc > (i+i_counter) && argv[i+i_counter][0] != '-') {
                            similar_limit = std::atoi(argv[i+i_counter]);
                            i_counter++;
                            if(argc > (i+i_counter) && argv[i+i_counter][0] != '-') {
                                dissimilar_limit = std::atoi(argv[i+i_counter]);
                            }
                        }
                        break;
                    case 'c':   //  closest cross
                        closest_cross = true;
                        i_counter++;
                        similar_limit = std::atoi(argv[i+i_counter]);
                        i_counter++;
                        dissimilar_limit = std::atoi(argv[i+i_counter]);
                        break;
                    default:
                        std::cout   << "Unrecognized option " << param << "\n"
                                    << "    add -h option for help\n";
                        return -1;
                        break;
                }
            }
            i += i_counter;
        } else {
            std::cout << "Unrecognized argument " << argv[i] << "\n"
                    "   add -h option for help\n";
            return -1;
        }
    }

	//	2.	Do the pair selection
	if(random_select) {
	    std::vector<std::pair<std::string, std::string> > images;
	    loadImagePaths(input_folder, file_extention, images);
		if(limit == 0) { //	no limit
			limit = images.size();
		    //  round the limit to 100
		    limit = limit - (limit % 100);
		}
		std::cout << "Random selection of size: " << limit << "\n";
		selectRandom(images, output_filename, limit);
	} else
	if(full_cross) {
        std::map<std::string, std::vector<std::string> > imagePaths;
        loadImagePathsByLabel(input_folder, file_extention, imagePaths);
        selectCross(imagePaths, output_filename, similar_limit, dissimilar_limit);
		return -1;
	} else
	if(closest_cross) {
	    std::map<std::string, std::vector<cv::Mat> > images;
	    int total_images = loadImagesByLabel(input_folder, file_extention, images);
	    std::map<std::string, std::vector<std::string> > imagePaths;
	    loadImagePathsByLabel(input_folder, file_extention, imagePaths);
	    selectClosest(images, imagePaths, output_filename, total_images, similar_limit, dissimilar_limit);
	}
}

