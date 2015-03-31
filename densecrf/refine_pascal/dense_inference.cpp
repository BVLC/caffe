/*
 * The code is modified from the NIPS demo code by Philipp Krähenbühl
 *
 *
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

#include "../libDenseCRF/densecrf.h"
#include "../libDenseCRF/util.h"
#include "dense_inference.h"
#include "../util/Timer.h"

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


// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];

unsigned int getColor( const unsigned char * c ){
  return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, unsigned int cc ){
  c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){
  unsigned char * r = new unsigned char[ W*H*3 ];
  for( int k=0; k<W*H; k++ ){
    int c = colors[ map[k] ];
    putColor( r+3*k, c );
  }
  return r;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
float * classify( const unsigned char * im, int W, int H, int M ){
  const float u_energy = -log( 1.0f / M );
  const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
  const float p_energy = -log( GT_PROB );
  float * res = new float[W*H*M];
  for( int k=0; k<W*H; k++ ){
    // Map the color to a label
    int c = getColor( im + 3*k );
    int i;
    for( i=0;i<nColors && c!=colors[i]; i++ );
    if (c && i==nColors){
      if (i<M)
	colors[nColors++] = c;
      else
	c=0;
    }
		
    // Set the energy
    float * r = res + k*M;
    if (c){
      for( int j=0; j<M; j++ )
	r[j] = n_energy;
      r[i] = p_energy;
    }
    else{
      for( int j=0; j<M; j++ )
	r[j] = u_energy;
    }
  }
  return res;
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
	unary[out_index] = -log(feat[in_index]);
      }
    }
  }
}

void OutputSetting(const InputData& inp) {
  std::cout << "Input Parameters: " << std::endl;
  std::cout << "ImgDir: "        << inp.ImgDir << std::endl;
  std::cout << "FeatureDir: "    << inp.FeatureDir << std::endl;
  std::cout << "SaveDir: "       << inp.SaveDir << std::endl;
  std::cout << "MaxIterations: " << inp.MaxIterations << std::endl;
  std::cout << "PosXStd: "  << inp.PosXStd << std::endl;
  std::cout << "PosYStd: "  << inp.PosYStd << std::endl;
  std::cout << "PosW: "     << inp.PosW    << std::endl;
  std::cout << "Bi_X_Std: " << inp.BilateralXStd << std::endl;
  std::cout << "Bi_Y_Std: " << inp.BilateralYStd << std::endl;
  std::cout << "Bi_R_Std: " << inp.BilateralRStd << std::endl;
  std::cout << "Bi_G_Std: " << inp.BilateralGStd << std::endl;
  std::cout << "Bi_B_Std: " << inp.BilateralBStd << std::endl;
  std::cout << "Bi_W: "     << inp.BilateralW    << std::endl;
}

int main( int argc, char* argv[]){
  InputData inp;
  inp.ImgDir = NULL;
  inp.FeatureDir = NULL;
  inp.SaveDir = NULL;
  inp.MaxIterations = 10;
  inp.PosXStd = 3;
  inp.PosYStd = 3;
  inp.PosW    = 3;
  inp.BilateralXStd = 60;
  inp.BilateralYStd = 60;
  inp.BilateralRStd = 20;
  inp.BilateralGStd = 20;
  inp.BilateralBStd = 20;
  inp.BilateralW    = 10;

  ParseInput(argc, argv, inp);
  OutputSetting(inp);

  assert(inp.ImgDir != NULL && inp.FeatureDir != NULL && inp.SaveDir != NULL);

  std::string pattern = "*.bin";
  std::vector<std::string> feat_file_names;
  std::string feat_folder(inp.FeatureDir);
  TraverseDirectory(feat_folder, pattern, false, feat_file_names);
  
  float* feat;
  float* unary;
  unsigned char* img;
  std::string fn;
  int img_row, img_col; //, img_channel;
  int feat_row, feat_col, feat_channel;

  CPrecisionTimer CTmr;
  CTmr.Start();
  for (size_t i = 0; i < feat_file_names.size(); ++i) {
    std::cout << "processing " << i << " (" << feat_file_names.size() << ")..." << std::endl;

    fn = std::string(inp.ImgDir) + "/" + feat_file_names[i] + ".ppm";
    img = readPPM(fn.c_str(), img_col, img_row);
    //LoadBinFile(fn, img, &img_row, &img_col, &img_channel);

    fn = std::string(inp.FeatureDir) + "/" + feat_file_names[i] + ".bin";
    LoadBinFile(fn, feat, &feat_row, &feat_col, &feat_channel);

    assert(img_row == feat_row && img_col == feat_col);

    unary = new float[feat_row*feat_col*feat_channel];
    ComputeUnaryForCRF(unary, feat, feat_row, feat_col, feat_channel);

    // Setup the CRF model
    DenseCRF2D crf(img_col, img_row, feat_channel);
    // Specify the unary potential as an array of size W*H*(#classes)
    // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ... (row-order)
    crf.setUnaryEnergy(unary);
    // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    crf.addPairwiseGaussian(inp.PosXStd, inp.PosYStd, inp.PosW);

    // add a color dependent term (feature = xyrgb)
    crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, inp.BilateralW);
	
    // Do map inference
    short* map = new short[img_row*img_col];
    crf.map(inp.MaxIterations, map);

    short* result = new short[img_row*img_col];
    ReshapeToMatlabFormat(result, map, img_row, img_col);

    // save results
    fn = std::string(inp.SaveDir) + "/" + feat_file_names[i] + ".bin";
    SaveBinFile(fn, result, feat_row, feat_col, 1);
    
    // delete
    delete[] result;
    delete[] unary;
    delete[] feat;
    delete[] img;
    delete[] map;
  }
  std::cout << "Time for inference: " << CTmr.Stop() << std::endl;
}
