#include <stdlib.h>
#include <math.h>



#ifdef COPYRIGHT_NOTICE

Copyright University of Amsterdam, 2002-2004. All rights reserved.

Contact person:
Jan-Mark Geusebroek (mark@science.uva.nl, http://www.science.uva.nl/~mark)
Intelligent Systems Lab Amsterdam
Informatics Institute, Faculty of Science, University of Amsterdam
Kruislaan 403, 1098 SJ Amsterdam, The Netherlands.


This software is being made available for individual research use only.
Any commercial use or redistribution of this software requires a license from
the University of Amsterdam.

You may use this work subject to the following conditions:

1. This work is provided "as is" by the copyright holder, with
absolutely no warranties of correctness, fitness, intellectual property
ownership, or anything else whatsoever.  You use the work
entirely at your own risk.  The copyright holder will not be liable for
any legal damages whatsoever connected with the use of this work.

2. The copyright holder retain all copyright to the work. All copies of
the work and all works derived from it must contain (1) this copyright
notice, and (2) additional notices describing the content, dates and
copyright holder of modifications or additions made to the work, if
any, including distribution and use conditions and intellectual property
claims.  Derived works must be clearly distinguished from the original
work, both by name and by the prominent inclusion of explicit
descriptions of overlaps and differences.

3. The names and trademarks of the copyright holder may not be used in
advertising or publicity related to this work without specific prior
written permission. 

4. In return for the free use of this work, you are requested, but not
legally required, to do the following:

- If you become aware of factors that may significantly affect other
  users of the work, for example major bugs or
  deficiencies or possible intellectual property issues, you are
  requested to report them to the copyright holder, if possible
  including redistributable fixes or workarounds.

- If you use the work in scientific research or as part of a larger
  software system, you are requested to cite the use in any related
  publications or technical documentation. The work is based upon:

    J. M. Geusebroek, A. W. M. Smeulders, and J. van de Weijer.
    Fast anisotropic gauss filtering. IEEE Trans. Image Processing,
    vol. 12, no. 8, pp. 938-943, 2003.
 
  related work:

    I.T. Young and L.J. van Vliet. Recursive implementation
    of the Gaussian filter. Signal Processing, vol. 44, pp. 139-151, 1995.
 
    B. Triggs and M. Sdika. Boundary conditions for Young-van Vliet
    recursive filtering. IEEE Trans. Signal Processing,
    vol. 54, pp. 2365-2367, 2006.
 
This copyright notice must be retained with all copies of the software,
including any modified or derived versions.

#endif /* COPYRIGHT_NOTICE */


#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979323846
#endif
#endif


/* define the input buffer type, e.g. "float" */
#define SRCTYPE double

/* define the output buffer type, should be at least "float" */
#define DSTTYPE double


/* the function prototypes */
void anigauss(SRCTYPE *input, DSTTYPE *output, int sizex, int sizey,
	double sigmav, double sigmau, double phi, int orderv, int orderu);
void YvVfilterCoef(double sigma, double *filter);
void TriggsM(double *filter, double *M);

static void f_iir_xline_filter(SRCTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter);
static void f_iir_yline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter);
static void f_iir_tline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter, double tanp);
static void f_iir_derivative_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double phi, int order);



/*
 *  the main function:
 *    anigauss(inbuf, outbuf, bufwidth, bufheight, sigma_v, sigma_u, phi,
 *       derivative_order_v, derivative_order_u);
 *
 *  v-axis = short axis
 *  u-axis = long axis
 *  phi = orientation angle in degrees
 *
 *  for example, anisotropic data smoothing:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 *  or, anisotropic edge detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 1, 0);
 *
 *  or, anisotropic line detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 2, 0);
 *
 *  or, in-place anisotropic data smoothing:
 *    anigauss(bufptr, bufptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 */


void anigauss(SRCTYPE *input, DSTTYPE *output, int sizex, int sizey,
	double sigmav, double sigmau, double phi, int orderv, int orderu)
{
    double	filter[7];
    double sigmax, sigmay, tanp;
    double su2, sv2;
    double phirad;
    double a11, a21, a22;
    int    i;

    su2 = sigmau*sigmau;
    sv2 = sigmav*sigmav;
    phirad = phi*PI/180.;

    a11 = cos(phirad)*cos(phirad)*su2 + sin(phirad)*sin(phirad)*sv2;
    a21 = cos(phirad)*sin(phirad)*(su2-sv2);
    a22 = cos(phirad)*cos(phirad)*sv2 + sin(phirad)*sin(phirad)*su2;

    sigmax = sqrt(a11-a21*a21/a22);
    tanp = a21/a22;
    sigmay = sqrt(a22);

    /* calculate filter coefficients of x-direction*/
    YvVfilterCoef(sigmax, filter);

    /* filter in the x-direction */
    f_iir_xline_filter(input,output,sizex,sizey,filter);

    /* calculate filter coefficients in tanp-direction */
    YvVfilterCoef(sigmay, filter);

    if (tanp != 0.0) {
        /* filter in the tanp-direction */
         f_iir_tline_filter(output,output,sizex,sizey,filter, tanp);
    }
    else {
        /* isotropic filter or anisotropic filter aligned with grid */
         f_iir_yline_filter(output,output,sizex,sizey,filter);
    }

    /* do the derivative filter: [-1,0,1] rotated over phi */
    for(i=0; i<orderv; i++)
        f_iir_derivative_filter(output, output, sizex, sizey, phirad-PI/2., 1);
    for(i=0; i<orderu; i++)
        f_iir_derivative_filter(output, output, sizex, sizey, phirad, 1);
}


void YvVfilterCoef(double sigma, double *filter)
{
    /* the recipe in the Young-van Vliet paper:
     * I.T. Young, L.J. van Vliet, M. van Ginkel, Recursive Gabor filtering.
     * IEEE Trans. Sig. Proc., vol. 50, pp. 2799-2805, 2002.
     *
     * (this is an improvement over Young-Van Vliet, Sig. Proc. 44, 1995)
     */

	double q, qsq;
    double scale;
    double B, b1, b2, b3;

	/* initial values */
	double m0 = 1.16680, m1 = 1.10783, m2 = 1.40586;
	double m1sq = m1*m1, m2sq = m2*m2;

	/* calculate q */
	if(sigma < 3.556)
		q = -0.2568 + 0.5784 * sigma + 0.0561 * sigma * sigma;
	else
		q = 2.5091 + 0.9804 * (sigma - 3.556);

	qsq = q*q;

	/* calculate scale, and b[0,1,2,3] */
	scale = (m0 + q) * (m1sq + m2sq + 2*m1*q + qsq);
	b1 = -q * (2*m0*m1 + m1sq + m2sq + (2*m0 + 4*m1) * q + 3*qsq) / scale;
	b2 = qsq * (m0 + 2*m1 + 3*q) / scale;
	b3 = - qsq * q / scale;
    
	/* calculate B */
	B = (m0 * (m1sq + m2sq))/scale;

	/* fill in filter */
    filter[0] = -b3;
    filter[1] = -b2;
    filter[2] = -b1;
    filter[3] = B;
    filter[4] = -b1;
    filter[5] = -b2;
    filter[6] = -b3;
}

void TriggsM(double *filter, double *M)
{
    double scale;
    double a1, a2, a3;

    a3 = filter[0];
    a2 = filter[1];
    a1 = filter[2];

    scale = 1.0/((1.0+a1-a2+a3)*(1.0-a1-a2-a3)*(1.0+a2+(a1-a3)*a3));
    M[0] = scale*(-a3*a1+1.0-a3*a3-a2);
    M[1] = scale*(a3+a1)*(a2+a3*a1);
    M[2] = scale*a3*(a1+a3*a2);
    M[3] = scale*(a1+a3*a2);
    M[4] = -scale*(a2-1.0)*(a2+a3*a1);
    M[5] = -scale*a3*(a3*a1+a3*a3+a2-1.0);
    M[6] = scale*(a3*a1+a2+a1*a1-a2*a2);
    M[7] = scale*(a1*a2+a3*a2*a2-a1*a3*a3-a3*a3*a3-a3*a2+a3);
    M[8] = scale*a3*(a1+a3*a2);
}


/**************************************
 * the low level filtering operations *
 **************************************/

/*
   all filters work in-place on the output buffer (DSTTYPE), except for the
   xline filter, which runs over the input (SRCTYPE)
   Note that the xline filter can also work in-place, in which case src=dst
   and SRCTYPE should be DSTTYPE
*/

static void
f_iir_xline_filter(SRCTYPE *src, DSTTYPE *dest, int sx, int sy, double *filter)
{
    int      i, j;
    double   b1, b2, b3;
    double   pix, p1, p2, p3;
    double   sum, sumsq;
    double   iplus, uplus, vplus;
    double   unp, unp1, unp2;
    double   M[9];

    sumsq = filter[3];
    sum = sumsq*sumsq;

    for (i = 0; i < sy; i++) {
		/* causal filter */
        b1 = filter[2]; b2 = filter[1]; b3 = filter[0];
		p1 = *src/sumsq; p2 = p1; p3 = p1;

        iplus = src[sx-1];
        for (j = 0; j < sx; j++) {
            pix = *src++ + b1*p1 + b2*p2 + b3*p3;
            *dest++ = pix;
            p3 = p2; p2 = p1; p1 = pix; /* update history */
        }

		/* anti-causal filter */

        /* apply Triggs border condition */
        uplus = iplus/(1.0-b1-b2-b3);
        b1 = filter[4]; b2 = filter[5]; b3 = filter[6];
        vplus = uplus/(1.0-b1-b2-b3);

        unp = p1-uplus;
        unp1 = p2-uplus;
        unp2 = p3-uplus;

        TriggsM(filter, M);

        pix = M[0]*unp+M[1]*unp1+M[2]*unp2 + vplus;
        p1  = M[3]*unp+M[4]*unp1+M[5]*unp2 + vplus;
        p2  = M[6]*unp+M[7]*unp1+M[8]*unp2 + vplus;
        pix *= sum; p1 *= sum; p2 *= sum;

        *(--dest) = pix;
        p3 = p2; p2 = p1; p1 = pix;

        for (j = sx-2; j >= 0; j--) {
            pix = sum * *(--dest) + b1*p1 + b2*p2 + b3*p3;
            *dest = pix;
            p3 = p2; p2 = p1; p1 = pix;
        }
        dest += sx;
	}
}

static void
f_iir_yline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy, double *filter)
{
    double   *p0, *p1, *p2, *p3, *pswap;
    double   *buf0, *buf1, *buf2, *buf3;
    double   *uplusbuf;
    int      i, j;
    double   b1, b2, b3;
    double   pix;
    double   sum, sumsq;
    double   uplus, vplus;
    double   unp, unp1, unp2;
    double   M[9];

    sumsq = filter[3];
    sum = sumsq*sumsq;

    uplusbuf = malloc(sx*sizeof(*uplusbuf));

    buf0 = malloc(sx*sizeof(*buf0));
    buf1 = malloc(sx*sizeof(*buf1));
    buf2 = malloc(sx*sizeof(*buf2));
    buf3 = malloc(sx*sizeof(*buf3));

    p0 = buf0; p1 = buf1; p2 = buf2; p3 = buf3;

    /* causal filter */
    b1 = filter[2]; b2 = filter[1]; b3 = filter[0];

    /* border first line*/
    for (j = 0; j < sx; j++) {
        pix = *src++/sumsq;
		p1[j] = pix; p2[j] = pix; p3[j] = pix;
    }
    /* calc last line for Triggs boundary condition */
    src += (sy-2)*sx;
    for (j = 0; j < sx; j++)
        uplusbuf[j] = *src++/(1.0-b1-b2-b3);
    src -= sy*sx;

    for (i = 0; i < sy; i++) {
        for (j = 0; j < sx; j++) {
            pix = *src++ + b1*p1[j] + b2*p2[j] + b3*p3[j];
            *dest++ = pix;
            p0[j] = pix;
        }

        /* shift history */
		pswap = p3; p3 = p2; p2 = p1; p1 = p0; p0 = pswap;
    }


    /* anti-causal filter */

    /* apply Triggs border condition */
    b1 = filter[4]; b2 = filter[5]; b3 = filter[6];
    TriggsM(filter, M);

    /* first line */
    p0 = uplusbuf;
    for (j = sx-1; j >= 0; j--) {
        uplus = p0[j];
        vplus = uplus/(1.0-b1-b2-b3);

        unp = p1[j]-uplus;
        unp1 = p2[j]-uplus;
        unp2 = p3[j]-uplus;
        pix = M[0]*unp+M[1]*unp1+M[2]*unp2 + vplus;
        pix *= sum;
        *(--dest) = pix;
        p1[j] = pix;
        pix  = M[3]*unp+M[4]*unp1+M[5]*unp2 + vplus;
        p2[j] = pix*sum;
        pix  = M[6]*unp+M[7]*unp1+M[8]*unp2 + vplus;
        p3[j] = pix*sum;
    }

    for (i = sy-2; i >= 0; i--) {
        for (j = sx-1; j >= 0; j--) {
            pix = sum * *(--dest) + b1*p1[j] + b2*p2[j] + b3*p3[j];
            *dest = pix;
            p0[j] = pix;
        }

        /* shift history */
		pswap = p3; p3 = p2; p2 = p1; p1 = p0; p0 = pswap;
    }

    free(buf0);
    free(buf1);
    free(buf2);
    free(buf3);
    free(uplusbuf);
}


static void
f_iir_tline_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double *filter, double tanp)
{
    double   *p0, *p1, *p2, *p3;
    double   *buf0, *buf1, *buf2, *buf3;
    double   *uplusbuf;
    int      i, j;
    double   b1, b2, b3;
    double   sum, sumsq;
    double   uplus, vplus;
    double   unp, unp1, unp2;
    double   M[9];
    double   pix, prev, val;
    double   res, prevres;
    double   xf;
    int      x;
    double   c, d;
    double   e, f;
    int      q4 = 0;

    /* check filter direction towards first or fourth quadrant */
    if (tanp <= 0.0) {
        q4 = 1;
        tanp = -tanp;
    }
    
    /* alloc buffer for Triggs boundary condition */
    uplusbuf = malloc(sx*sizeof(*uplusbuf));

    /* alloc recursion line buffers */
    buf0 = malloc((sx+sy*tanp+2)*sizeof(*buf0));
    buf1 = malloc((sx+sy*tanp+2)*sizeof(*buf1));
    buf2 = malloc((sx+sy*tanp+2)*sizeof(*buf2));
    buf3 = malloc((sx+sy*tanp+2)*sizeof(*buf3));

    if (q4) {
        buf0 += (int)(sy*tanp+1);
        buf1 += (int)(sy*tanp+1);
        buf2 += (int)(sy*tanp+1);
        buf3 += (int)(sy*tanp+1);
    }

    sumsq = filter[3];
    sum = sumsq*sumsq;

    /* causal filter */
    b1 = filter[2]; b2 = filter[1]; b3 = filter[0];

    /* first line */
    p1 = buf1; p2 = buf2; p3 = buf3;
    for (j = 0; j < sx; j++) {
        pix = *src++;
        *dest++ = pix; *p1++ = pix; *p2++ = pix; *p3++ = pix;
    }

    /* calc last line for Triggs boundary condition */
    src += (sy-2)*sx;
    for (j = 0; j < sx; j++)
        uplusbuf[j] = *src++ * sumsq/(1.0-b1-b2-b3);
    src -= (sy-1)*sx;

    x = 0;
    for (i = 1; i < sy; i++) {
        xf = i*tanp;

        /* border handling at image corner */
        if (q4) {
            p1 = buf1-x; p2 = buf2-x; p3 = buf3-x;
            for (j=1; j <= (int)(xf)-x; j++) {
                p1[-j] = p1[0];
                p2[-j] = p2[0];
                p3[-j] = p3[0];
            }
        }
        else {
            p1 = buf1+x; p2 = buf2+x; p3 = buf3+x;
            for (j=1; j <= (int)(xf)-x; j++) {
                p1[sx+j-1] = p1[sx-1];
                p2[sx+j-1] = p2[sx-1];
                p3[sx+j-1] = p3[sx-1];
            }
        }

        /* calc interpolation coefficients */
        x = (int)xf;
        c = xf-(double)x;
        d = 1.0-c;

        e = c; f = d;
        if (!q4) {
            res = d; d = c; c = res;
            res = f; f = e; e = res;
        }

        c *= sumsq; d *= sumsq;

        /* set buffers at start */
        if (q4) {
            p0 = buf0-x; p1 = buf1-x; p2 = buf2-x; p3 = buf3-x;
        }
        else {
            p0 = buf0+x; p1 = buf1+x; p2 = buf2+x; p3 = buf3+x;
        }
        prev = *src;
        prevres = sumsq*prev + b1 * *p1 + b2 * *p2 + b3 * *p3;

        /* run the filter */
        for (j = 0; j < sx; j++) {
            pix = *src++;
            val = c*pix+d*prev;
            prev = pix;

            res = val + b1 * *p1++ + b2 * *p2++ + b3 * *p3++;
            *p0++ = res;
            *dest++ = f*res+e*prevres;
            prevres = res;
        }

        /* shift history */
        p0 = buf3; buf3 = buf2; buf2 = buf1; buf1 = buf0; buf0 = p0;
    }

    /* anti-causal */

    /* apply Triggs border condition */
    b1 = filter[4]; b2 = filter[5]; b3 = filter[6];
    TriggsM(filter, M);

    /* first line */
    x = (int)((sy-1)*tanp);
    if (q4) {
        p1 = buf1+sx-x; p2 = buf2+sx-x; p3 = buf3+sx-x;
    }
    else {
        p1 = buf1+sx+x; p2 = buf2+sx+x; p3 = buf3+sx+x;
    }
    p0 = uplusbuf+sx;
    for (j = 0; j < sx; j++) {
        uplus = *(--p0);
        vplus = uplus/(1.0-b1-b2-b3);

        unp = *(--p1)-uplus;
        unp1 = *(--p2)-uplus;
        unp2 = *(--p3)-uplus;
        pix = M[0]*unp+M[1]*unp1+M[2]*unp2 + vplus;
        pix *= sumsq;
        *(--dest) = pix; *p1 = pix;
        pix  = M[3]*unp+M[4]*unp1+M[5]*unp2 + vplus;
        *p2 = pix*sumsq;
        pix  = M[6]*unp+M[7]*unp1+M[8]*unp2 + vplus;
        *p3 = pix*sumsq;
    }

    for (i = sy-2; i >= 0; i--) {
        xf = i*tanp;

        /* border handling at image corner */
        if (q4) {
            p1 = buf1-x; p2 = buf2-x; p3 = buf3-x;
            for (j=1; j <= x-(int)(xf); j++) {
                p1[sx+j-1] = p1[sx-1];
                p2[sx+j-1] = p2[sx-1];
                p3[sx+j-1] = p3[sx-1];
            }
        }
        else {
            p1 = buf1+x; p2 = buf2+x; p3 = buf3+x;
            for (j=1; j <= x-(int)(xf); j++) {
                p1[-j] = p1[0];
                p2[-j] = p2[0];
                p3[-j] = p3[0];
            }
        }

        /* calc interpolation coefficients */
        x = (int)xf;
        c = xf-(double)x;
        d = 1.0-c;

        e = c; f = d;
        c *= sumsq; d *= sumsq;

        if (!q4) {
            res = d; d = c; c = res;
            res = f; f = e; e = res;
        }

        /* set buffers at start */
        if (q4) {
            p0 = buf0+sx-x; p1 = buf1+sx-x; p2 = buf2+sx-x; p3 = buf3+sx-x;
        }
        else {
            p0 = buf0+sx+x; p1 = buf1+sx+x; p2 = buf2+sx+x; p3 = buf3+sx+x;
        }
        prev = *(dest-1);
        prevres = sumsq*prev + b1 * *(p1-1) + b2 * *(p2-1) + b3 * *(p3-1);

        /* run the filter */
        for (j = 0; j < sx; j++) {
            pix = *(--dest);
            val = d*pix+c*prev;
            prev = pix;

            res = val + b1 * *(--p1) + b2 * *(--p2) + b3 * *(--p3);
            *(--p0) = res;
            *dest = e*res+f*prevres;
            prevres = res;
        }

        /* shift history */
        p0 = buf3; buf3 = buf2; buf2 = buf1; buf1 = buf0; buf0 = p0;
    }

    if (q4) {
        buf0 -= (int)(sy*tanp+1);
        buf1 -= (int)(sy*tanp+1);
        buf2 -= (int)(sy*tanp+1);
        buf3 -= (int)(sy*tanp+1);
    }
    free(buf0);
    free(buf1);
    free(buf2);
    free(buf3);
    free(uplusbuf);
}

/* rotated [-1,0,1] derivative filter */
static void
f_iir_derivative_filter(DSTTYPE *src, DSTTYPE *dest, int sx, int sy,
    double phi, int order)
{
   int				i, j;
   DSTTYPE   *prev, *center, *next;
   DSTTYPE   *buf, *pstore;
   double  pn, pc, pp;
   double  cn, cc, cp;
   double  nn, nc, np;
   double  cosp, sinp;

   buf = malloc(sx*sizeof(*buf));

   sinp = 0.5*sin(phi); cosp = 0.5*cos(phi);

   center = src; prev = src; next = src+sx;
   for (i = 0; i < sy; i++) {
        pstore = buf;
        pn = *prev++; cn = *center++; nn = *next++;
        pp = pn; pc = pn;
        cp = cn; cc = cn;
        np = pn; nc = nn;
        *pstore++ = cc;
        for (j = 1; j < sx; j++) {
            pn = *prev++;
            cn = *center++;
            nn = *next++;
            *dest++ = sinp*(pc-nc)+cosp*(cn-cp);
            pp = pc; pc = pn;
            cp = cc; cc = cn;
            np = pc; nc = nn;
            *pstore++ = cc;
        }
        *dest++ = sinp*(pc-nc)+cosp*(cn-cp);
        prev = buf;
        if (i==sy-2)
            next -= sx;
    }

   free(buf);
}
