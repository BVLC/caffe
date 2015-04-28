/*
 * The code is modified from the NIPS demo code by Philipp Krähenbühl
 *
 * Support LoadMatFile and SaveMatFile. 
 *
 *
 */
/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>

#include "matio.h"

#include "../libDenseCRF/densecrf.h"
#include "../libDenseCRF/util.h"
#include "../util/Timer.h"

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, const int row, const int col,
		 int* channel = NULL, bool do_ppm_format = false);

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, const int row, const int col, 
		 int* channel, bool do_ppm_format) {
  mat_t *matfp;
  matfp = Mat_Open(fn.c_str(), MAT_ACC_RDONLY);
  if (matfp == NULL) {
    std::cerr << "Error opening MAT file " << fn;
  }

  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp,"data");
  if (matvar == NULL) {
    std::cerr << "Field 'data' not present in MAT file " << fn << std::endl;
  }

  if (matvar->class_type != matio_class_map<T>()) {
    std::cerr << "Field 'data' must be of the right class (single/double) in MAT file " << fn << std::endl;
  }
  if (matvar->rank >= 4) {
    if (matvar->dims[3] != 1) {
      std::cerr << "Rank: " << matvar->rank << ". Field 'data' cannot have ndims > 3 in MAT file " << fn << std::endl;
    }
  }

  int file_size = 1;
  int data_size = row * col;
  for (int k = 0; k < matvar->rank; ++k) {
    file_size *= matvar->dims[k];
    
    if (k > 1) {
      data_size *= matvar->dims[k];
    }
  }

  assert(data_size <= file_size);

  T* file_data = new T[file_size];
  data = new T[data_size];
  
  int ret = Mat_VarReadDataLinear(matfp, matvar, file_data, 0, 1, file_size);
  if (ret != 0) {
    std::cerr << "Error reading array 'data' from MAT file " << fn << std::endl;
  }

  // matvar->dims[0] : width
  // matvar->dims[1] : height
  int in_offset = matvar->dims[0] * matvar->dims[1];
  int in_ind, out_ind;
  int data_channel = static_cast<int>(matvar->dims[2]);

  // extract from file_data
  if (do_ppm_format) {
    int out_offset = col * data_channel;

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
	  out_ind = m * out_offset + n * data_channel + c;

	  // perform transpose of file_data
	  in_ind  = n + m * matvar->dims[1];  

	  // note the minus sign
	  data[out_ind] = -file_data[in_ind + c*in_offset];  
	}
      }
    }
  } else {
    int out_offset = row * col;

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
	  in_ind  = m + n * matvar->dims[1];
	  out_ind = m + n * row; 
	  data[out_ind + c*out_offset] = -file_data[in_ind + c*in_offset];	  
	}
      }
    }
  }

  if(channel != NULL) {
    *channel = data_channel;
  }  


  Mat_VarFree(matvar);
  Mat_Close(matfp);

  delete[] file_data;
}


template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
      int* row = NULL, int* col = NULL, int* channel = NULL);

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
      int row = 1, int col = 1, int channel = 1);

template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
    int* row, int* col, int* channel) {
  //data.clear();

  std::ifstream ifs(fn.c_str(), std::ios_base::in | std::ios_base::binary);

  if(!ifs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }

  int num_row, num_col, num_channel;

  ifs.read((char*)&num_row, sizeof(int));
  ifs.read((char*)&num_col, sizeof(int));
  ifs.read((char*)&num_channel, sizeof(int));

  int num_el;

  num_el = num_row * num_col * num_channel;

  //data.resize(num_el);
  data = new T[num_el];

  ifs.read((char*)&data[0], sizeof(T)*num_el);

  ifs.close();

  if(row!=NULL) {
    *row = num_row;
  }

  if(col!=NULL) {
    *col = num_col;
  }
 
  if(channel != NULL) {
    *channel = num_channel;
  }

}

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
    int row, int col, int channel) {
  std::ofstream ofs(fn.c_str(), std::ios_base::out | std::ios_base::binary);

  if(!ofs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }  

  ofs.write((char*)&row, sizeof(int));
  ofs.write((char*)&col, sizeof(int));
  ofs.write((char*)&channel, sizeof(int));

  int num_el;

  num_el = row * col * channel;

  ofs.write((char*)&data[0], sizeof(T)*num_el);

  ofs.close();
}


void TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames) {
  DIR *dir, *tstdp;
  struct dirent *dp;

  //open the directory
  if((dir  = opendir(path.c_str())) == NULL) {
    std::cout << "Error opening " << path << std::endl;
    return;
  }

  while ((dp = readdir(dir)) != NULL) {
    tstdp=opendir(dp->d_name);
		
    if(tstdp) {
      closedir(tstdp);
      if(subdirectories) {
	//TraverseDirectory(
      }
    } else {
      if(fnmatch(pattern.c_str(), dp->d_name, 0)==0) {
	//std::string tmp(path);	
	//tmp.append("/").append(dp->d_name);
	//fileNames.push_back(tmp);  //assume string ends with .bin

	std::string tmp(dp->d_name);
	fileNames.push_back(tmp.substr(0, tmp.length()-4));

	//std::cout << fileNames.back() << std::endl;
      }
    }
  }

  closedir(dir);
  return;
}


struct InputData {
  char* ImgDir;
  char* FeatureDir;
  char* SaveDir;
  int MaxIterations;
  float PosXStd;
  float PosYStd;
  float PosW;
  float BilateralXStd;
  float BilateralYStd;
  float BilateralRStd;
  float BilateralGStd;
  float BilateralBStd;
  float BilateralW;

};

int ParseInput(int argc, char** argv, struct InputData& OD) {
  for(int k=1;k<argc;++k) {
    if(::strcmp(argv[k], "-id")==0 && k+1!=argc) {
      OD.ImgDir = argv[++k];
    } else if(::strcmp(argv[k], "-fd")==0 && k+1!=argc) {
      OD.FeatureDir = argv[++k];
    } else if(::strcmp(argv[k], "-sd")==0 && k+1!=argc) {
      OD.SaveDir = argv[++k];
    } else if(::strcmp(argv[k], "-i")==0 && k+1!=argc) {
      OD.MaxIterations = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-px")==0 && k+1!=argc) {
      OD.PosXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-py")==0 && k+1!=argc) {
      OD.PosYStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-pw")==0 && k+1!=argc) {
      OD.PosW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bx")==0 && k+1!=argc) {
      OD.BilateralXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-by")==0 && k+1!=argc) {
      OD.BilateralYStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bw")==0 && k+1!=argc) {
      OD.BilateralW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-br")==0 && k+1!=argc) {
      OD.BilateralRStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bg")==0 && k+1!=argc) {
      OD.BilateralGStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bb")==0 && k+1!=argc) {
      OD.BilateralBStd = atof(argv[++k]);
    } 
  }
  return 0;
}

void ReshapeToMatlabFormat(short*& result, short* map, int img_row, int img_col) {
  //row-order to column-order

  int out_index, in_index;

  for (int h = 0; h < img_row; ++h) {
    for (int w = 0; w < img_col; ++w) {
      out_index = w * img_row + h;
      in_index  = h * img_col + w;
      result[out_index] = map[in_index];
    }
  }
}

void ComputeUnaryForCRF(float*& unary, float* feat, int feat_row, int feat_col, int feat_channel) {
  int out_index, in_index;

  for (int h = 0; h < feat_row; ++h) {
    for (int w = 0; w < feat_col; ++w) {
      for (int c = 0; c < feat_channel; ++c) {
	out_index = (h * feat_col + w) * feat_channel + c;
	in_index  = (c * feat_col + w) * feat_row + h;
	////unary[out_index] = -log(feat[in_index]);
	unary[out_index] = -feat[in_index];
      }
    }
  }
}

void GetImgNamesFromFeatFiles(std::vector<std::string>& out, const std::vector<std::string>& in, const std::string& strip_pattern) {
  for (size_t k = 0; k < in.size(); ++k) {
    size_t pos = in[k].find(strip_pattern);
    if (pos != std::string::npos) {
      out.push_back(in[k].substr(0, pos));      
    }
  }
}

void OutputSetting(const InputData& inp) {
  std::cout << "Input Parameters: " << std::endl;
  std::cout << "ImgDir:           " << inp.ImgDir << std::endl;
  std::cout << "FeatureDir:       " << inp.FeatureDir << std::endl;
  std::cout << "SaveDir:          " << inp.SaveDir << std::endl;
  std::cout << "MaxIterations:    " << inp.MaxIterations << std::endl;
  //std::cout << "MaxImgSize:       " << inp.MaxImgSize << std::endl;
  //std::cout << "NumClass:         " << inp.NumClass << std::endl;
  std::cout << "PosW:      " << inp.PosW    << std::endl;
  std::cout << "PosXStd:   " << inp.PosXStd << std::endl;
  std::cout << "PosYStd:   " << inp.PosYStd << std::endl;
  std::cout << "Bi_W:      " << inp.BilateralW    << std::endl;
  std::cout << "Bi_X_Std:  " << inp.BilateralXStd << std::endl;
  std::cout << "Bi_Y_Std:  " << inp.BilateralYStd << std::endl;
  std::cout << "Bi_R_Std:  " << inp.BilateralRStd << std::endl;
  std::cout << "Bi_G_Std:  " << inp.BilateralGStd << std::endl;
  std::cout << "Bi_B_Std:  " << inp.BilateralBStd << std::endl;  
}

int main( int argc, char* argv[]){
  InputData inp;
  // default values
  inp.ImgDir = NULL;
  inp.FeatureDir = NULL;
  inp.SaveDir = NULL;

  inp.MaxIterations = 10;

  inp.BilateralW    = 5;
  inp.BilateralXStd = 70;
  inp.BilateralYStd = 70;
  inp.BilateralRStd = 5;
  inp.BilateralGStd = 5;
  inp.BilateralBStd = 5;

  inp.PosW    = 3;
  inp.PosXStd = 3;
  inp.PosYStd = 3;

  ParseInput(argc, argv, inp);
  OutputSetting(inp);

  assert(inp.ImgDir != NULL && inp.FeatureDir != NULL && inp.SaveDir != NULL);
  
  std::string pattern = "*.mat";
  std::vector<std::string> feat_file_names;
  std::string feat_folder(inp.FeatureDir);

  TraverseDirectory(feat_folder, pattern, false, feat_file_names);
  
  std::string strip_pattern("_blob_0");
  std::vector<std::string> img_file_names;
  GetImgNamesFromFeatFiles(img_file_names, feat_file_names, strip_pattern);

  float* feat;
  unsigned char* img;
  std::string fn;
  int feat_row, feat_col, feat_channel;

  bool do_ppm_format = true;

  CPrecisionTimer CTmr;
  CTmr.Start();
  for (size_t i = 0; i < feat_file_names.size(); ++i) {
    if ( (i+1) % 100 == 0) {
      std::cout << "processing " << i << " (" << feat_file_names.size() << ")..." << std::endl;
    }

    fn = std::string(inp.ImgDir) + "/" + img_file_names[i] + ".ppm";
    img = readPPM(fn.c_str(), feat_col, feat_row);

    fn = std::string(inp.FeatureDir) + "/" + feat_file_names[i] + ".mat";
    LoadMatFile(fn, feat, feat_row, feat_col, &feat_channel, do_ppm_format);

    // Setup the CRF model
    DenseCRF2D crf(feat_col, feat_row, feat_channel);
    // Specify the unary potential as an array of size W*H*(#classes)
    // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ... (row-order)
    crf.setUnaryEnergy(feat);
    // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    crf.addPairwiseGaussian(inp.PosXStd, inp.PosYStd, inp.PosW);

    // add a color dependent term (feature = xyrgb)
    crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, inp.BilateralW);
	
    // Do map inference
    short* map = new short[feat_row*feat_col];
    crf.map(inp.MaxIterations, map);

    short* result = new short[feat_row*feat_col];
    ReshapeToMatlabFormat(result, map, feat_row, feat_col);

    // save results
    fn = std::string(inp.SaveDir) + "/" + img_file_names[i] + ".bin";
    SaveBinFile(fn, result, feat_row, feat_col, 1);
    
    // delete
    delete[] result;
    delete[] feat;
    delete[] img;
    delete[] map;
  }
  std::cout << "Time for inference: " << CTmr.Stop() << std::endl;

}
