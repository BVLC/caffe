FROM yi/caffe:gpu

MAINTAINER Igor Rabkin <igor.rabkin@xiaoyi.com>

ARG TFLOW=tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl

################################################
#     Basic desktop environment                #
################################################

# Locale, language
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8


#################################################
#          Set Time Zone Asia/Jerusalem         #
#################################################

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


################################
# Updating PIP and Dependences #
################################

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    ipykernel \
    jupyter \
    matplotlib \
    numpy \
    scipy \
    sklearn \
    pandas \
    && \
    python -m ipykernel.kernelspec

###################################
# Install TensorFlow GPU version. #
###################################

  RUN cd /
  ARG CRED="server:123server123"
  RUN  curl -u ${CRED} ftp://yifileserver/DOCKER_IMAGES/Tensorflow/Current/${TFLOW} -o ${TFLOW} && \
       pip --no-cache-dir install ${TFLOW} && \
       rm -f ${TFLOW}


##################################################
# Configure the build for our CUDA configuration #
##################################################

ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=5.2,6.1
ENV TF_CUDA_VERSION=8.0
ENV TF_CUDNN_VERSION=6.0

################ INTEL MKL SUPPORT #################

ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN cd /usr/local/lib && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libiomp5.so -o libiomp5.so && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libmklml_gnu.so -o libmklml_gnu.so && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libmklml_intel.so -o libmklml_intel.so

####################################################


#########################################
# Add Welcome Message With Instructions #
#########################################

RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' \
	>> /etc/bash.bashrc \
	; echo "\
||||||||||||||||||||||||||||||||||||||||||||||||||\n\
|                                                |\n\
| Docker container running Ubuntu                |\n\
| with TensorFlow ${TF_BRANCH} optimized for CPU        |\n\
| with Intel(R) MKL Support                      |\n\
|                                                |\n\
||||||||||||||||||||||||||||||||||||||||||||||||||\n\
\n "\
	> /etc/motd


#####################
# Standard SSH Port #
#####################

EXPOSE 22


#####################
# Default command   #
#####################

ENTRYPOINT ["/tini", "--"]
CMD ["/usr/sbin/sshd", "-D"]
RUN ["/bin/bash"]
