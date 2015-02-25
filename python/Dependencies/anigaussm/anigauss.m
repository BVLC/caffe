% anigauss - Recursive anisotropic Gauss filtering
% Usage:
%   out = anigauss(in, sigma_v, sigma_u, phi,
%     derivative_order_v, derivative_order_u);
% 
%   v-axis = short axis
%   u-axis = long axis
%   phi = orientation angle in degrees
% 
%   parameters sigma_u, phi, and derivative_order_{v,w} are optional.
%   sigma_u defaults to the value of sigma_v (isotropic filtering),
%   phi defaults to zero degrees,
%   derivative orders default to 0 (no differentiation, only smooth data).
%
%   Note that for isotropic filtering a slightly faster algorithm is used than
%   for anisotropic filtering. Furthermore, execution time depends on the order
%   of differentiation. Note that the execution time is independend of the
%   values for sigma.
%
% Examples:
%
%   isotropic filtering:
%   a=zeros(512,512);
%   a(256,256)=1; 
%   tic;c=anigauss(a,10);toc
%   elapsed_time =
%      0.0500
%
%   anisotropic filtering:
%   a=zeros(512,512);
%   a(256,256)=1; 
%   tic;c=anigauss(a,10,3,30);toc
%   elapsed_time =
%      0.0600
%
% Usage:
%
%   isotropic data smoothing:
%     out = anigauss(in, 3.0);
%
%   isotropic data differentiation along y-axis:
%     out = anigauss(in, 3.0, 3.0, 0.0, 0, 1);
%
%   anisotropic data smoothing:
%     out = anigauss(in, 3.0, 7.0, 30.0);
% 
%   anisotropic edge detection:
%     out = anigauss(in, 3.0, 7.0, 30.0, 1, 0);
% 
%   anisotropic line detection:
%     out = anigauss(in, 3.0, 7.0, 30.0, 2, 0);
% 
%
%
% Copyright University of Amsterdam, 2002-2004. All rights reserved.
% 
% Contact person:
% Jan-Mark Geusebroek (mark@science.uva.nl, http://www.science.uva.nl/~mark)
% Intelligent Systems Lab Amsterdam
% Informatics Institute, Faculty of Science, University of Amsterdam
% Kruislaan 403, 1098 SJ Amsterdam, The Netherlands.
% 
% 
% This software is being made available for individual research use only.
% Any commercial use or redistribution of this software requires a license from
% the University of Amsterdam.
% 
% You may use this work subject to the following conditions:
% 
% 1. This work is provided "as is" by the copyright holder, with
% absolutely no warranties of correctness, fitness, intellectual property
% ownership, or anything else whatsoever.  You use the work
% entirely at your own risk.  The copyright holder will not be liable for
% any legal damages whatsoever connected with the use of this work.
% 
% 2. The copyright holder retain all copyright to the work. All copies of
% the work and all works derived from it must contain (1) this copyright
% notice, and (2) additional notices describing the content, dates and
% copyright holder of modifications or additions made to the work, if
% any, including distribution and use conditions and intellectual property
% claims.  Derived works must be clearly distinguished from the original
% work, both by name and by the prominent inclusion of explicit
% descriptions of overlaps and differences.
% 
% 3. The names and trademarks of the copyright holder may not be used in
% advertising or publicity related to this work without specific prior
% written permission. 
% 
% 4. In return for the free use of this work, you are requested, but not
% legally required, to do the following:
% 
% - If you become aware of factors that may significantly affect other
%   users of the work, for example major bugs or
%   deficiencies or possible intellectual property issues, you are
%   requested to report them to the copyright holder, if possible
%   including redistributable fixes or workarounds.
% 
% - If you use the work in scientific research or as part of a larger
%   software system, you are requested to cite the use in any related
%   publications or technical documentation. The work is based upon:
% 
%     J. M. Geusebroek, A. W. M. Smeulders, and J. van de Weijer.
%     Fast anisotropic gauss filtering. IEEE Trans. Image Processing,
%     vol. 12, no. 8, pp. 938-943, 2003.
%
%   related work:
%  
%     I.T. Young and L.J. van Vliet. Recursive implementation
%     of the Gaussian filter. Signal Processing, vol. 44, pp. 139-151, 1995.
%  
%     B. Triggs and M. Sdika. Boundary conditions for Young-van Vliet
%     recursive filtering. IEEE Trans. Signal Processing,
%     vol. 54, pp. 2365-2367, 2006.
%  
% This copyright notice must be retained with all copies of the software,
% including any modified or derived versions.
