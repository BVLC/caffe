namespace caffe {

class KernelParameters {
 public:
  double r, g, b;
  double lambda, sigma;
  double omega, phi, theta;

  KernelParameters()
      : r(1), g(1), b(1), lambda(1), sigma(0.6), omega(M_PI), phi(0), theta(0) {
  }

  void generate(int kernelId, int numberOfKernels, int kernelSize) {
    if ((numberOfKernels == 32) && (kernelSize == 5))
      generateCifarLike(kernelId, kernelSize);
    else if ((numberOfKernels == 64) && (kernelSize == 7))
      generateGoogleNetLike(kernelId, kernelSize);
    else if ((numberOfKernels == 64) && (kernelSize == 3))
      generateVggLike(kernelId, kernelSize);
    else if ((numberOfKernels == 96) && (kernelSize == 11))
      generateAlexNetLike(kernelId, kernelSize);
    else
      LOG(FATAL) << "No predefined gabor filters for this topology.";
  }

  void generateCifarLike(int kernelId, int kernelSize) {
    lambda = 0.5;

    if (kernelId < 8) {
      omega = M_PI * (kernelSize - 1) / 2 / 1;
      theta = (kernelId % 8) * M_PI / 8;
    } else if (kernelId < 14) {
      omega = M_PI * (kernelSize - 1) / 2 / 2;
      theta = ((kernelId - 2) % 6) * M_PI / 6 + M_PI / 12;
    } else if (kernelId < 16) {
      lambda = 0.5;
      sigma = 0.75;
      omega = M_PI * (kernelSize - 1) / 2 / 8;
      phi = (kernelId % 2) * M_PI;
      r = 1;
      g = -1;
      b = 1;
    } else {
      omega = M_PI * (kernelSize - 1) / 2 / 4;
      theta = (kernelId % 4) * M_PI / 2 + M_PI / 4 + M_PI / 8;
      phi = M_PI / 2;
    }

    if (kernelId >= 30) {
      theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 + M_PI / 8;
      r = 1;
      g = 1;
      b = 0;
    } else if (kernelId >= 28) {
      theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 - M_PI / 8;
      r = -1;
      g = 1;
      b = -1;
    } else if (kernelId >= 24) {
      r = -1;
      g = 1;
      b = 1;
    } else if (kernelId >= 20) {
      r = 1;
      g = 0;
      b = -1;
    }
  }

  void generateGoogleNetLike(int kernelId, int kernelSize) {
    if (kernelId < 32) {
      int rotation = kernelId / 8;
      int frequency = kernelId % 8;
      int phase = kernelId % 2;
      lambda = 1 / (1 + frequency / 8.);
      sigma = 0.4 + 0.2 * frequency / 8;
      omega = M_PI * (kernelSize - 1) / 2 / (1 + frequency / 2.);
      phi = phase * M_PI + M_PI * 12 / 32;
      theta = rotation * M_PI / 4;
    } else if (kernelId < 40) {
      sigma = 0.45;
      lambda = 0.5;
      omega = M_PI * (kernelSize - 1) / 2;
      theta = (kernelId % 8) * M_PI / 8;
      phi = M_PI;
    } else if (kernelId < 46) {
      int phase = (kernelId - 1) / 3;
      int size = (kernelId - 1) % 3;
      lambda = 1 / (1 + size / 2.);
      sigma = 1. / (2.5 - size / 2.);
      omega = M_PI / 4;
      phi = phase * M_PI;
      r = 0.25;
      g = -1;
      b = 1;
    } else if (kernelId < 48) {
      lambda = 2. / 3;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 12;
      theta = (kernelId % 8) * M_PI + M_PI / 8;
      phi = M_PI / 2;
      r = 1;
      g = 0.1;
      b = -0.5;
    } else if (kernelId < 56) {
      lambda = 2. / 3;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 12;
      theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
      phi = M_PI / 2;
      r = -0.5;
      g = 0.1;
      b = 1;
    } else if (kernelId < 60) {
      lambda = 1;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 12;
      theta = (kernelId % 8) * M_PI / 2 + M_PI / 8;
      phi = M_PI / 2;
      r = 0.25;
      g = -1;
      b = 1;
    } else {
      lambda = 2. / 3;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 12;
      theta = (kernelId % 8) * M_PI / 2 + M_PI / 8;
      phi = M_PI / 2;
      r = -1;
      g = -1;
      b = 1;
    }
  }

  void generateVggLike(int kernelId, int kernelSize) {
    generateGoogleNetLike(kernelId, kernelSize);
    sigma = 1;
  }

  void generateAlexNetLike(int kernelId, int kernelSize) {
    lambda = 1. / 3;

    if (kernelId < 48) {
      int rotation = kernelId / 8;
      int frequency = kernelId % 8;
      int phase = kernelId % 2;
      lambda /= (1 + frequency / 8.);
      sigma = 0.5 + 0.2 * frequency / 8;
      omega = M_PI * (kernelSize - 1) / 2 / (1 + frequency / 2.);
      phi = phase * M_PI + M_PI * 12 / 32;
      theta = rotation * M_PI / 6;
    } else if (kernelId < 56) {
      int phase = kernelId / 4;
      int size = kernelId % 4;
      lambda /= (1 + size / 2.);
      sigma = 1. / (2.5 - size / 2.);
      omega = M_PI / 4;
      phi = phase * M_PI;
      r = 0.25;
      g = -1;
      b = 1;
    } else if (kernelId < 60) {
      lambda /= 1.5;
      sigma = 0.75;
      omega = M_PI * (kernelSize - 1) / 2 / 8;
      theta = (kernelId % 4) * M_PI / 2 + M_PI / 8;
      phi = M_PI / 2;
      r = -1;
      g = 1;
      b = -0.5;
    } else if (kernelId < 64) {
      lambda /= 3;
      sigma = 2;
      omega = M_PI * (kernelSize - 1) / 2 / 8;
      theta = (kernelId % 4) * M_PI / 2 + M_PI / 8;
      phi = M_PI / 2;
      r = 1;
      g = -0.5;
      b = -0.75;
    } else if (kernelId < 72) {
      lambda /= 1.5;
      sigma = 0.75;
      omega = M_PI * (kernelSize - 1) / 2 / 4;
      theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
      phi = M_PI / 2;
      r = 1;
      g = 0.1;
      b = -0.75;
    } else if (kernelId < 80) {
      lambda /= 1.5;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 12;
      theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
      phi = M_PI / 2;
      r = -0.5;
      g = 0.1;
      b = 1;
    } else if (kernelId < 88) {
      lambda /= 2.5;
      sigma = 1;
      omega = M_PI * (kernelSize - 1) / 2 / 8;
      theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
      phi = M_PI / 2;
      r = -1;
      g = -1;
      b = 1;
    } else if (kernelId < 92) {
      omega = M_PI * (kernelSize - 1) / 2 / 16;
      theta = (kernelId % 4) * M_PI / 2 + M_PI / 16;
      phi = M_PI / 2;
    } else {
      lambda /= 4;
      sigma = 0.75;
      omega = M_PI * (kernelSize - 1) / 2 / 4;
      theta = (kernelId % 8) * M_PI / 4 + M_PI / 32;
      phi = M_PI / 2;
      r = -1;
      g = -1;
      b = 1;
    }
  }
};

template <typename Dtype> class KernelGenerator {
 public:
  KernelGenerator(int numberOfKernels, int kernelSize)
      : numberOfKernels(numberOfKernels), kernelSize(kernelSize),
        kernels(new Dtype[getNumberOfElements()]) {}

  ~KernelGenerator() { delete[] kernels; }

  void generate() {
    for (int kernelId = 0; kernelId < numberOfKernels; kernelId++)
      generateKernel(kernelId);
  }

  const Dtype *getKernelData() const { return kernels; }

  int getSizeOfKernelData() const {
    return getNumberOfElements();
  }

 private:
  int numberOfKernels;
  int kernelSize;
  Dtype *kernels;

  int getNumberOfElements() const {
    return numberOfKernels * 3 * kernelSize * kernelSize;
  }

  void generateKernel(int kernelId) {
    KernelParameters param;
    param.generate(kernelId, numberOfKernels, kernelSize);

    for (int ky = 0; ky < kernelSize; ky++)
      for (int kx = 0; kx < kernelSize; kx++) {
        double x = 2. * kx / (kernelSize - 1) - 1;
        double y = 2. * ky / (kernelSize - 1) - 1;

        double dis = exp(-(x * x + y * y) / (2 * param.sigma * param.sigma));
        double arg = x * cos(param.theta) - y * sin(param.theta);
        double per = cos(arg * param.omega + param.phi);
        double val = param.lambda * dis * per;

        kernels[kx + kernelSize * (ky + kernelSize * (0 + 3 * kernelId))] =
            (Dtype)(param.r * val);
        kernels[kx + kernelSize * (ky + kernelSize * (1 + 3 * kernelId))] =
            (Dtype)(param.g * val);
        kernels[kx + kernelSize * (ky + kernelSize * (2 + 3 * kernelId))] =
            (Dtype)(param.b * val);
      }
  }
};
};  // namespace caffe
