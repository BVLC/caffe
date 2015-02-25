/*
   The Matlab mex function.
   If necessary to recompile, type:
       mex -v -g anigauss_mex.c anigauss.c
   from within matlab.
   For windows platforms, you may want to use the provided "anigauss.dll" file.
*/


#include "mex.h"

extern void anigauss(double *input, double *output, int sizex, int sizey,
	double sigmav, double sigmau, double phi, int orderv, int orderu);

void mexFunction(int nlhs,mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    double *in, *out;
    double sigmav, sigmau, phi = 0.0;
    int    orderv = 0, orderu = 0;
    int    m, n;

	/*
	 * Check the input arguments and the output argument
	 */
    if ((nrhs<2) || (nrhs>6) || (nrhs==5) || (nlhs!=1))
        mexErrMsgTxt(
            "use: out = anigauss(in, sigmav, sigmau, phi, orderv, orderu);");

	if ( mxGetNumberOfDimensions(prhs[0]) != 2 ) 
		{ mexErrMsgTxt("anigauss: input array should be of dimension 2"); }

    if (nrhs>=2) {
        in = mxGetPr(prhs[0]); 
        sigmav = mxGetScalar(prhs[1]);
        sigmau = sigmav;
    }
    if (nrhs>=3)
        sigmau = mxGetScalar(prhs[2]);
    if (nrhs>=4)
        phi = mxGetScalar(prhs[3]);
    if (nrhs==6) {
        orderv = (int)(mxGetScalar(prhs[4])+0.5);
        orderu = (int)(mxGetScalar(prhs[5])+0.5);
    }

    if ((orderv<0) || (orderu<0))
		{ mexErrMsgTxt("anigauss: derivative orders should be positive"); }

    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);

	/* pointers to output array */

	plhs[0]=mxCreateDoubleMatrix(m, n, mxREAL );	
	if ( plhs[0] == NULL )
        { mexErrMsgTxt("No more memory for out array"); }
	out = (double *)mxGetPr( plhs[0] );
	
	anigauss(in, out, m, n, sigmav,  sigmau,  phi-90.0, orderv,  orderu);
}
