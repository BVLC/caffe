#include <cmath>
#include "mex.h"
#include "segment-image.h"

#define UInt8 char


/*
 * Segment an image
 *
 * Matlab Wrapper around the code of Felzenszwalb and Huttenlocher created 
 * by Jasper Uijlings, 2012
 *
 * Returns a color image representing the segmentation. 
 * JASPER: Random is replaced by just an index.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
double *segment_image_index(image<rgb> *im, float sigma, float c, int min_size,
			  int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;
 
  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
	edges[num].a = y * width + x;
	edges[num].b = y * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
	num++;
      }

      if (y < height-1) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y < height-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y > 0)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width*height, num, edges, c);
  
  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  //image<rgb> *output = new image<rgb>(width, height);

  // pick random colors for each component
  double *colors = new double[width*height];
  for (int i = 0; i < width*height; i++)
    colors[i] = 0;
  
  int idx = 1;
  double* indexmap = new double[width * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      if (!(colors[comp])){
          colors[comp] = idx;
          idx = idx + 1;
      }

      //imRef(output, x, y) = colors[comp];
      indexmap[x * height + y] = colors[comp];
    }
  }  
  //mexPrintf("indexmap 0: %f\n", indexmap[0]);
  //mexPrintf("indexmap 1: %f\n", indexmap[1]);

  delete [] colors;
  delete u;

  return indexmap;
}

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[])
{
    // Checking number of arguments
    if(nlhs > 3){
        mexErrMsgTxt("Function has three return values");
        return;
    }

    if(nrhs != 4){
        mexErrMsgTxt("Usage: mexFelzenSegment(UINT8 im, double sigma, double c, int minSize)");
        return;
    }

    if(!mxIsClass(input[0], "uint8")){
        mexErrMsgTxt("Only image arrays of the UINT8 class are allowed.");
        return;
    }

    // Load in arrays and parameters
    UInt8* matIm = (UInt8*) mxGetPr(input[0]);
    int nrDims = (int) mxGetNumberOfDimensions(input[0]);
    int* dims = (int*) mxGetDimensions(input[0]);
    double* sigma = mxGetPr(input[1]);
    double* c = mxGetPr(input[2]);
    double* minSize = mxGetPr(input[3]);
    int min_size = (int) *minSize;

    int height = dims[0];
    int width = dims[1];
    int imSize = height * width;
    
    // Convert to image. 
    int idx;
    image<rgb>* theIm = new image<rgb>(width, height);
    for (int x = 0; x < width; x++){
        for (int y = 0; y < height; y++){
            idx = x * height + y;
            imRef(theIm, x, y).r = matIm[idx];
            imRef(theIm, x, y).g = matIm[idx + imSize];
            imRef(theIm, x, y).b = matIm[idx + 2 * imSize];
        }
    }

    // KOEN: Disable randomness of the algorithm
    srand(12345);

    // Call Felzenswalb segmentation algorithm
    int num_css;
    //image<rgb>* segIm = segment_image(theIm, *sigma, *c, min_size, &num_css);
    double* segIndices = segment_image_index(theIm, *sigma, *c, min_size, &num_css);
    //mexPrintf("numCss: %d\n", num_css);

    // The segmentation index image
    out[0] = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
    double* outSegInd = mxGetPr(out[0]);

    // Keep track of minimum and maximum of each blob
    out[1] = mxCreateDoubleMatrix(num_css, 4, mxREAL);
    double* minmax = mxGetPr(out[1]);
    for (int i=0; i < num_css; i++)
        minmax[i] = dims[0];
    for (int i= num_css; i < 2 * num_css; i++)
        minmax[i] = dims[1];

    // Keep track of neighbouring blobs using square matrix
    out[2] = mxCreateDoubleMatrix(num_css, num_css, mxREAL);
    double* nn = mxGetPr(out[2]);

    // Copy the contents of segIndices
    // Keep track of neighbours
    // Get minimum and maximum
    // These actually comprise of the bounding boxes
    double currDouble;
    int mprev, curr, prevHori, mcurr;
    for(int x = 0; x < width; x++){
        mprev = segIndices[x * height]-1;
        for(int y=0; y < height; y++){
            //mexPrintf("x: %d y: %d\n", x, y);
            idx = x * height + y;
            //mexPrintf("idx: %d\n", idx);
            //currDouble = segIndices[idx]; 
            //mexPrintf("currDouble: %d\n", currDouble);
            curr = segIndices[idx]; 
            //mexPrintf("curr: %d\n", curr);
            outSegInd[idx] = curr; // copy contents
            //mexPrintf("outSegInd: %f\n", outSegInd[idx]);
            mcurr = curr-1;

            // Get neighbours (vertical)
            //mexPrintf("idx: %d", curr * num_css + mprev);
            //mexPrintf(" %d\n", curr + num_css * mprev);
            //mexPrintf("mprev: %d\n", mprev);
            nn[(mcurr) * num_css + mprev] = 1;
            nn[(mcurr) + num_css * mprev] = 1;

            // Get horizontal neighbours
            //mexPrintf("Get horizontal neighbours\n");
            if (x > 0){
                prevHori = outSegInd[(x-1) * height + y] - 1;
                nn[mcurr * num_css + prevHori] = 1;
                nn[mcurr + num_css * prevHori] = 1;
            }

            // Keep track of min and maximum index of blobs
            //mexPrintf("Keep track of min and maximum index\n");
            if (minmax[mcurr] > y)
                minmax[mcurr] = y;
            if (minmax[mcurr + num_css] > x)
                minmax[mcurr + num_css] = x;
            if (minmax[mcurr + 2 * num_css] < y)
                minmax[mcurr + 2 * num_css] = y;
            if (minmax[mcurr + 3 * num_css] < x)
                minmax[mcurr + 3 * num_css] = x;

            //mexPrintf("Mprev = mcurr");
            mprev = mcurr;
        }
    }

    // Do minmax plus one for Matlab
    for (int i=0; i < 4 * num_css; i++)
        minmax[i] += 1;

    delete theIm;
    delete [] segIndices;

    return;
}
       







