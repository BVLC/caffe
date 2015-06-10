/** \file create_rt_imageset.cpp
 * \brief executable to create rigid transformed variations of labeled images
 * into an imageset.
 *
 * Loads the paths to a labeled set of images; Each folder is a label and all
 * the images contained in that folder belongs to that label.
 * A set of image pairs and their similarity label are selected based on their
 * Euclidean distance.
 * To each of these images a rigid transform is applied and stored in a
 * subfolder.
 * All of these images are then stored in an imageset file.
 *
 * \Notes:
 * Based on code from the Caffe examples and tools
 *
 * \version
 * -    v0.1a   Initial version
 *
 * Future versions:
 *
 * \author      Floris Gaisser <f.gaisser@tudelft.nl>
 * \date        v0.1a   2015-06-09 ~ 2015-06-?
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

//  C/C++
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <vector>

//  Google logging
#include "glog/logging.h"
#include "google/protobuf/text_format.h"

//  Caffe
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "leveldb/db.h"
#include <lmdb.h>

//  OpenCV
#include <opencv2/opencv.hpp>

/*
 * load image images grouped by label
 * ToDo: Move to some more general location
 */
int loadImagesByLabel(
        std::string input_folder,
        std::string file_extention,
        std::map<   std::string,
                    std::vector<
                        std::pair<
                            std::vector<std::string>,
                            cv::Mat> > > &images) {

    int total_images = 0;
    //  check for trailing '/'
    if(input_folder.find_last_of("/") != input_folder.size()-1)
        input_folder.append("/");
    if(!file_extention.empty() && file_extention.find(".") != 0) {
        file_extention = "." + file_extention;
    }

    //  check dir
    struct stat info;
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
                std::cout << "loading images from label: " << input_dir_contents_handle->d_name << "\n";
                images[input_dir_contents_handle->d_name] = std::vector<std::pair<std::vector<std::string>, cv::Mat> >();
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
                    images[input_dir_contents_handle->d_name].push_back(
                            std::make_pair(
                                    std::vector<std::string>(),
                                    cv::imread(filename, cv::IMREAD_GRAYSCALE)));
                    images[input_dir_contents_handle->d_name].back().first.push_back(filename);
//                    std::cout << filename << "\n";
//                    cv::imshow("input", images[input_dir_contents_handle->d_name].back().second);
//                    cv::waitKey(1);
                    if(images[input_dir_contents_handle->d_name].back().second.empty()) {
                        std::cerr << "help! empty image: " << filename << "\n";
                    }
                    total_images++;

                    //  create the rt folder and the numbered folder containing the RT-ed images
                    std::string rt_folder = input_folder;
                    rt_folder.append(input_dir_contents_handle->d_name);
                    rt_folder.append("/rt/");
                    if(stat(rt_folder.c_str(), &info ) != 0) {
                        CHECK_EQ(mkdir(rt_folder.c_str(), 0744), 0)
                            << "mkdir " << rt_folder << " failed";
                    }
                    rt_folder.append(
                            std::string(label_dir_contents_handle->d_name).substr(
                                    0,
                                    std::string(label_dir_contents_handle->d_name).find_last_of('.', std::string::npos)));
                    rt_folder.append("/");
                    if(stat(rt_folder.c_str(), &info ) != 0) {
                        CHECK_EQ(mkdir(rt_folder.c_str(), 0744), 0)
                            << "mkdir " << rt_folder << " failed";
                    }

                    //  Do the RT and save
                    float translation_step_size_    = 3;
                    float rotation_step_size_       = 0.05*3.14159;
                    float scale_step_size_          = 0.01;
                    float perspective_step_size_    = 0.02/28;

                    int translation_steps_          = 2;
                    int rotation_steps_             = 0;
                    int scale_steps_                = 0;
                    int perspective_steps_          = 0;

                    cv::Mat rt_image;
                    float cx = (float)images[input_dir_contents_handle->d_name].back().second.cols/2.;
                    float cy = (float)images[input_dir_contents_handle->d_name].back().second.rows/2.;
        //          std::cout << "cx: " << cx << "\tcy: " << cy << "\n";
                    //  start transform loop
                    cv::Mat H = cv::Mat::eye(3, 3, CV_32FC1);
                    for(int tx = -translation_steps_; tx <= translation_steps_; tx++) {
                        for(int ty = -translation_steps_; ty <= translation_steps_; ty++) {
                            float H02 = translation_step_size_ * tx;
                            float H12 = translation_step_size_ * ty;
                            for(int r = -rotation_steps_; r <= rotation_steps_; r++) {
                                float sin_r =  std::sin(rotation_step_size_ * r);
                                float cos_r =  std::cos(rotation_step_size_ * r);
                                H.at<float>(0, 0) = cos_r;
                                H.at<float>(1, 1) = cos_r;
                                //  roatate about center [cx, cy]
                                H.at<float>(0, 2) = H02 + (sin_r * cx) - ((1 - cos_r) * cy);
                                H.at<float>(1, 2) = H12 + ((1 - cos_r) * cx) - (sin_r * cy);
                                for(int s = -scale_steps_; s <= scale_steps_; s++) {
                                    H.at<float>(2, 2) = 1.0 + scale_step_size_ * s;
                                    for(int px = -perspective_steps_; px <= perspective_steps_; px++) {
                                        px = 0;
                                        for(int py = -perspective_steps_; py <= perspective_steps_; py++) {
                                            H.at<float>(2, 0) = perspective_step_size_ * px;
                                            H.at<float>(2, 1) = perspective_step_size_ * py;
                                            H.at<float>(1, 0) =  sin_r + 2 * perspective_step_size_ * px;
                                            H.at<float>(0, 1) = -sin_r + 2 * perspective_step_size_ * py;
                                            //  do the magic here
                                            //H = cv::Mat::eye(3, 3, CV_32FC1);
        //                                  std::cout   << "tx:\t" << (translation_step_size_ * tx) << "\n"
        //                                              << "ty:\t" << (translation_step_size_ * ty) << "\n"
        //                                              << "r:\t" << (rotation_step_size_ * r) << "\n"
        //                                              << "s:\t" << (scale_step_size_ * s) << "\n"
        //                                              << "px:\t" << (perspective_step_size_ * px) << "\n"
        //                                              << "py:\t" << (perspective_step_size_ * py) << "\n";
        //                                  std::cout << H << "\n";
                                            std::stringstream ss;
                                            ss << rt_folder;
                                            ss << "t_" << tx << "-" << ty << "_r_" << r << "_s_" << s << "_p_" << px << "-" << py << ".pgm";

                                            images[input_dir_contents_handle->d_name].back().second.copyTo(rt_image);
                                            cv::warpPerspective(
                                                    rt_image,
                                                    rt_image, H,
                                                    cv::Size( rt_image.cols, rt_image.rows),
                                                    cv::INTER_LINEAR,
                                                    cv::BORDER_CONSTANT,
                                                    cv::Scalar(128));
                                            cv::imwrite(ss.str(), rt_image);

                                            //  add the path of this (new) rt image
                                            images[input_dir_contents_handle->d_name].back().first.push_back(ss.str());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return total_images;
}

bool compare_pairs(std::pair<std::vector<std::string>, double> a, std::pair<std::vector<std::string>, double> b) {
    return (a.second < b.second);
}

bool checkAddForLabeledPair(
        int i,
        std::string label_i,
        int j,
        std::string label_j,
        std::vector<std::string> &pair_hashes) {
    std::stringstream ss, ss_reverse;
    ss  << label_i << "-" << i << "_" << label_j << "-" << j;
    ss_reverse << label_j << "-" << j << "_" << label_i << "-" << i;
    if(std::find(pair_hashes.begin(), pair_hashes.end(), ss.str()) == pair_hashes.end()) {
        // not found, so add
        pair_hashes.push_back(ss.str());
        pair_hashes.push_back(ss_reverse.str());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    //  0.  vars
    int             similar_limit = 0;
    int             dissimilar_limit = 0;
    std::string     output_filename = "";
    std::string     input_folder = "";
    std::string     file_extention = "";

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
                    case 'e':   //  input folder
                        i_counter++;
                        file_extention = std::string(argv[i+i_counter]);
                        break;
                    case 'c':   //  closest cross
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

    //  get the images!
    std::map<std::string, std::vector<std::pair<std::vector<std::string>, cv::Mat> > > images;
    loadImagesByLabel(input_folder, file_extention, images);

    //  check the limits
    if(similar_limit <= 0) {
        similar_limit = 5;
    }
    if(dissimilar_limit <= 0) {
        dissimilar_limit = 5;
    }
    if(similar_limit > 100 || dissimilar_limit > 100) {
        std::cerr << "limits are to high!\n";
        return -1;
    }

    //  create the output file
    std::ofstream output_file(output_filename.c_str(), std::ios::out | std::ios::binary);
    CHECK(output_file) << "Unable to open file " << output_filename;

    cv::Mat curr_image, match_image;
    std::vector<std::pair<std::pair<std::string, std::string>, int> > selected_image_pairs;
    std::vector<std::string> pair_hashes;
    std::map<std::string, std::vector<std::pair<std::vector<std::string>, cv::Mat> > >::iterator label_it, other_label_it;
    //  loop over the labels
    for(label_it = images.begin(); label_it != images.end(); label_it++) {
        //  loop over the images
        std::cout   << "Start processing label: " << label_it->first << ".\n";
        for(int image_counter = 0; image_counter < label_it->second.size(); image_counter++) {
            //  find closest similar images
            curr_image = label_it->second.at(image_counter).second;
            std::vector<std::pair<std::vector<std::string>, double> > closest;
            for(int similar_counter = 0; similar_counter < label_it->second.size(); similar_counter++) {
                if(similar_counter == image_counter) { continue; }  // don't match to yourself
                match_image = label_it->second.at(similar_counter).second;

                double distance = cv::norm(curr_image, match_image, cv::NORM_L1);

                closest.push_back(std::make_pair(
                        label_it->second.at(similar_counter).first,
                        distance));

                std::sort(closest.begin(), closest.end(), compare_pairs);
                if(closest.size() > similar_limit)
                    closest.pop_back();
            }

            //  looped over all similar images, found most similar images:
            //  ToDo: How to check for doubles?
            for(int match_counter = 0; match_counter < similar_limit; match_counter++) {
                for(int i = 0; i < label_it->second.at(image_counter).first.size(); i++) {
                    for(int j = 0; j < closest.at(match_counter).first.size(); j++) {
                        selected_image_pairs.push_back(
                                std::make_pair(
                                        std::make_pair(
                                                label_it->second.at(image_counter).first.at(i),
                                                closest.at(match_counter).first.at(j)),
                                        1));
                    }
                }
            }
            //  Do RT on the similar images
            //  Add these to the selected_pairs

            //  find closest dissimilar images
            closest.clear();
            for(other_label_it = images.begin(); other_label_it != images.end(); other_label_it++) {
                if(label_it->first == other_label_it->first) { continue; } // skip if the labels are the same!
                for(int dissimilar_counter = 0; dissimilar_counter < other_label_it->second.size(); dissimilar_counter++) {
                    match_image = other_label_it->second.at(dissimilar_counter).second;
                    double distance = cv::norm(curr_image, match_image, cv::NORM_L1);

                    closest.push_back(std::make_pair(
                            other_label_it->second.at(dissimilar_counter).first,
                            distance));

                    std::sort(closest.begin(), closest.end(), compare_pairs);
                    if(closest.size() > dissimilar_limit)
                        closest.pop_back();
                }
            }

            //  looped over all dissimilar images, found most dissimilar images:
            for(int match_counter = 0; match_counter < dissimilar_limit; match_counter++) {
                for(int i = 0; i < label_it->second.at(image_counter).first.size(); i++) {
                    for(int j = 0; j < closest.at(match_counter).first.size(); j++) {
                        selected_image_pairs.push_back(
                                std::make_pair(
                                        std::make_pair(
                                                label_it->second.at(image_counter).first.at(i),
                                                closest.at(match_counter).first.at(j)),
                                        0));
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

