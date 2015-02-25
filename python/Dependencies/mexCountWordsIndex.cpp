#include <cmath>
#include "mex.h"
void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[])
{
    // Checking number of arguments
    if (nlhs > 2){
        mexErrMsgTxt("Error: function has only two output parameters");
        return;
    }

    if (nrhs != 4){
        mexErrMsgTxt("Error: Needs exactly two four input parameters");
        return;
    }

    int numWords = (int) mxGetScalar(input[3]);
    int numIndices = (int) mxGetScalar(input[2]);

    // Load in arrays
    double* indices = mxGetPr(input[0]);
    double* a = mxGetPr( input[1] );
    int aNum = (int) mxGetNumberOfElements(input[1]);
    int totIndices = (int) mxGetNumberOfElements(input[0]); // number of elements. Not confuse with max
    int numLoops = aNum / totIndices;

    // Create output histogram
    out[0] = mxCreateDoubleMatrix(numIndices, numWords, mxREAL);
    double* histogram = mxGetPr(out[0]);
    //histogram = histogram - 1;

    out[1] = mxCreateDoubleMatrix(numIndices, 1, mxREAL);
    double* count = mxGetPr(out[1]);

    double* aP = a;
    int iPval;
    for(int j=0; j < numLoops; j++){
        double* iP = indices;
        for(int i=0;i < totIndices; i++){
            //mexPrintf("%d\n", i);
            if (*aP){
                //(*(histogram + (((int) *aP) -1) * numIndices + ((int) *iP - 1)))++;
                //count++;
                iPval = ((int) *iP) -1;
                histogram[(((int) *aP) - 1) * numIndices + iPval]++;
                count[iPval]++;
            }

            //arrayI = (int) *aP;
            //histogram[arrayI]++;
            aP++;
            iP++;
        }
    }

    return;
}
