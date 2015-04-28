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
#include <cmath>

#include "../libDenseCRF/densecrf.h"
#include "../libDenseCRF/util.h"

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

int main( int argc, char* argv[]){
  if (argc<4){
    printf("Usage: %s image annotations output\n", argv[0] );
    return 1;
  }
  // Number of labels
  const int M = 21;
  // Load the color image and some crude annotations (which are used in a simple classifier)
  int W, H, GW, GH;
  unsigned char * im = readPPM( argv[1], W, H );
  if (!im){
    printf("Failed to load image!\n");
    return 1;
  }
  unsigned char * anno = readPPM( argv[2], GW, GH );
  if (!anno){
    printf("Failed to load annotations!\n");
    return 1;
  }
  if (W!=GW || H!=GH){
    printf("Annotation size doesn't match image!\n");
    return 1;
  }
	
  /////////// Put your own unary classifier here! ///////////
  float * unary = classify( anno, W, H, M );
  ///////////////////////////////////////////////////////////
	
  // Setup the CRF model
  DenseCRF2D crf(W, H, M);
  // Specify the unary potential as an array of size W*H*(#classes)
  // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ... (row-order)
  crf.setUnaryEnergy( unary );
  // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
  // x_stddev = 3
  // y_stddev = 3
  // weight = 3
  crf.addPairwiseGaussian( 3, 3, 3 );
  // add a color dependent term (feature = xyrgb)
  // x_stddev = 60
  // y_stddev = 60
  // r_stddev = g_stddev = b_stddev = 20
  // weight = 10
  crf.addPairwiseBilateral( 60, 60, 20, 20, 20, im, 10 );
	
  // Do map inference
  short * map = new short[W*H];
  crf.map(10, map);
	
  // Store the result
  unsigned char *res = colorize( map, W, H );
  writePPM( argv[3], W, H, res );
	
  delete[] im;
  delete[] anno;
  delete[] res;
  delete[] map;
  delete[] unary;
}
