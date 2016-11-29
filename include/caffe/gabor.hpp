namespace caffe {

class KernelParameters
{
    public:

        double r, g, b;
        double lambda, sigma;
        double omega, phi, theta;

        KernelParameters() :
            r(1), g(1), b(1),
            lambda(1), sigma(0.6),
            omega(M_PI), phi(0), theta(0)
        {
        }

        void generateCifarLike(int kernelId, int kernelSize)
        {
            if(kernelId < 8) {
                omega = M_PI * (kernelSize - 1) / 2 / 1;
                theta = (kernelId % 8) * M_PI / 8;
            }
            else if(kernelId < 16) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 8) * M_PI / 8 + M_PI / 32;
            }
            else {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 4) * M_PI / 2 + M_PI / 4 + M_PI / 8;
                phi = M_PI / 2;
            }

            if(kernelId >= 30) {
                theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 + M_PI / 8;
                r = 1;
                g = 1;
                b = 0;
            }
            else if(kernelId >= 28) {
                theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 - M_PI / 8;
                r = -1;
                g = 1;
                b = 1;
            }
            else if(kernelId >= 24) {
                r = -1;
                g = 1;
                b = -1;
            }
            else if(kernelId >= 20) {
                r = 1;
                g = 0;
                b = -1;
            }
        }

        void generateGoogleNetLike(int kernelId, int kernelSize)
        {
            if(kernelId < 8) {
                omega = M_PI * (kernelSize - 1) / 2 / 1;
                theta = (kernelId % 8) * M_PI / 8 + M_PI / 32;
            }
            else if(kernelId < 16) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 8) * M_PI / 8 + M_PI / 32;
            }
            else if(kernelId < 24) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
            }
            else if(kernelId < 32) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
            }
            else if(kernelId < 36) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 4) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
                r = 1;
                g = 0;
                b = -1;
            }
            else if(kernelId < 40) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 4) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
                r = 0;
                g = 1;
                b = -1;
            }
            else if(kernelId < 48) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
                r = 1;
                g = -1;
                b = -1;
            }
            else if(kernelId < 56) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
                r = -1;
                g = 1;
                b = -1;
            }
            else {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 16;
                phi = M_PI / 2;
                r = -1;
                g = -1;
                b = 1;
            }
        }

        void generateVggLike(int kernelId, int kernelSize)
        {
            generateGoogleNetLike(kernelId, kernelSize);
            sigma = 1;
        }

        void generateAlexNetLike(int kernelId, int kernelSize)
        {
            if(kernelId < 8) {
                omega = M_PI * (kernelSize - 1) / 2 / 1;
                theta = (kernelId % 8) * M_PI / 8 + M_PI / 32;
            }

            else if(kernelId < 24) {
                omega = M_PI * (kernelSize - 1) / 2 / 2;
                theta = (kernelId % 16) * M_PI / 16 + M_PI / 128;
            }

            else if(kernelId < 32) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 8 + M_PI / 128;
            }

            else if(kernelId < 40) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
            }

            else if(kernelId < 48) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
            }

            else if(kernelId < 56) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
                r = 1;
                g = -1;
                b = -1;
            }

            else if(kernelId < 64) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
                r = -1;
                g = 1;
                b = -1;
            }

            else if(kernelId < 72) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
                r = -1;
                g = -1;
                b = 1;
            }

            else if(kernelId < 80) {
                omega = M_PI * (kernelSize - 1) / 2 / 4;
                theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
                phi = M_PI / 2;
                r = -1;
                g = 0;
                b = 1;
            }

            else if(kernelId < 84) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 4) * M_PI / 2 + M_PI / 16;
                phi = M_PI / 2;
                r = -1;
                g = 1;
                b = 0;
            }

            else if(kernelId < 88) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 4) * M_PI / 2 + M_PI / 16;
                phi = M_PI / 2;
                r = 1;
                g = 1;
                b = 0;
            }

            else if(kernelId < 92) {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 4) * M_PI / 2 + M_PI / 16;
                phi = M_PI / 2;
                r = 0;
                g = -1;
                b = 1;
            }

            else {
                omega = M_PI * (kernelSize - 1) / 2 / 8;
                theta = (kernelId % 4) * M_PI / 2 + M_PI / 16;
                phi = M_PI / 2;
                r = 1;
                g = 0;
                b = -1;
            }
        }
};

template <typename Real>
class KernelGenerator
{
    public:

        KernelGenerator(int numberOfKernels, int kernelSize) :
            numberOfKernels(numberOfKernels), kernelSize(kernelSize),
            kernels(new Real[getNumberOfElements()])
        {
        }

        ~KernelGenerator()
        {
            delete [] kernels;
        }

        void generate(double lambda)
        {
            for(int kernelId = 0; kernelId < numberOfKernels; kernelId++)
                generateKernel(kernelId, lambda);
        }

        const Real *getKernelData() const
        {
            return kernels;
        }

        int getSizeOfKernelData() const
        {
            return sizeof(Real) * getNumberOfElements();
        }


    private:

        int numberOfKernels;
        int kernelSize;
        Real *kernels;

        int getNumberOfElements() const
        {
            return numberOfKernels * 3 * kernelSize * kernelSize;
        }

        void generateKernel(int kernelId, double lambda)
        {
            KernelParameters param;
            getKernelParams(param, kernelId);

            for(int ky = 0; ky < kernelSize; ky++)
                for(int kx = 0; kx < kernelSize; kx++) {
                    double x = 2. * kx / (kernelSize - 1) - 1;
                    double y = 2. * ky / (kernelSize - 1) - 1;

                    double dis = exp(-(x * x + y * y) / (2 * param.sigma * param.sigma));
                    double arg = x * cos(param.theta) - y * sin(param.theta);
                    double per = cos(arg * param.omega + param.phi);
                    double val = lambda * dis * per;

                    kernels[kx + kernelSize * (ky + kernelSize * (0 + 3 * kernelId))] = (Real) (param.r * val);
                    kernels[kx + kernelSize * (ky + kernelSize * (1 + 3 * kernelId))] = (Real) (param.g * val);
                    kernels[kx + kernelSize * (ky + kernelSize * (2 + 3 * kernelId))] = (Real) (param.b * val);
                }
        }

        void getKernelParams(KernelParameters &param, int kernelId)
        {
            if((numberOfKernels == 32) && (kernelSize == 5))
                param.generateCifarLike(kernelId, kernelSize);
            else if((numberOfKernels == 64) && (kernelSize == 7))
                param.generateGoogleNetLike(kernelId, kernelSize);
            else if((numberOfKernels == 64) && (kernelSize == 3))
                param.generateVggLike(kernelId, kernelSize);
            else if((numberOfKernels == 96) && (kernelSize == 11))
                param.generateAlexNetLike(kernelId, kernelSize);
            else
                LOG(FATAL) << "No predefined gabor filters for this topology.";
        }
};

// EXAMPLE USAGE:
// KernelGenerator kernelGenerator(numberOfKernels, kernelSize);
// kernelGenerator.generate(lambda);
// memcpy(dest, kernelGenerator.getKernelData(), kernelGenerator.getSizeOfKernelData());
}; //namespace cafffe
