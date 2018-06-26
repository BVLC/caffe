FROM yi/caffe:gpu
 
LABEL MAINTAINER="Igor Rabkin<igor.rabkin@xiaoyi.com>"

################################################
#          Basic desktop environment           #
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

#################################################
#     Very basic installations                  #
#################################################

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -q install -y -o Dpkg::Options::="--force-confnew" --no-install-recommends \
    python-software-properties \
    software-properties-common \
    python-dev \
    build-essential \
    curl \
    git \
    iputils-ping \
    zip \
    unzip \
    tree \
    nano \
    tzdata \
    mlocate \
    vim \
    sudo \
    pkg-config && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    

#################################################
# PID 1 - signal forwarding and zombie fighting #
#################################################

# Add Tini
ARG TINI_VERSION=v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini


##################################
# Install JDK 8 (latest edition) #
##################################

RUN apt-get -q update &&\
    DEBIAN_FRONTEND="noninteractive" apt-get -q install -y -o Dpkg::Options::="--force-confnew" --no-install-recommends software-properties-common &&\
    add-apt-repository -y ppa:openjdk-r/ppa &&\
    apt-get -q update &&\
    DEBIAN_FRONTEND="noninteractive" apt-get -q install -y -o Dpkg::Options::="--force-confnew" --no-install-recommends openjdk-8-jre-headless &&\
    apt-get -q clean -y && rm -rf /var/lib/apt/lists/* && rm -f /var/cache/apt/*.bin
	
	
##############################################################
# Upgrade packages on image & Installing and Configuring SSH #
##############################################################

RUN apt-get -q update &&\
    DEBIAN_FRONTEND="noninteractive" apt-get -q upgrade -y -o Dpkg::Options::="--force-confnew" --no-install-recommends &&\
    DEBIAN_FRONTEND="noninteractive" apt-get -q install -y -o Dpkg::Options::="--force-confnew" --no-install-recommends openssh-server &&\
    rm -rf /var/lib/apt/lists/* && rm -f /var/cache/apt/*.bin 
	
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir /var/run/sshd 

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile	


#################################
# Set user jenkins to the image #
#################################

RUN useradd -m -d /home/jenkins -s /bin/bash jenkins &&\
    echo "jenkins:jenkins" | chpasswd
	
# Add the jenkins user to sudoers
RUN echo "jenkins  ALL=(ALL)  NOPASSWD: ALL" >> /etc/sudoers

# Set full permission for jenkins folder
RUN chmod -R 777 /home/jenkins


##################################################################
#                Pick up some TF dependencies                    #
##################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    emacs \
    golang \
    python-dev 
         
################################	
# Updating PIP and Dependences #
################################

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py	
 
RUN pip --no-cache-dir install --upgrade \ 
       pip setuptools    
RUN pip --no-cache-dir install wheel
	
###################################
# Install TensorFlow GPU version. #
###################################

  RUN cd /
  ARG TFLOW=tensorflow-1.7.1-cp27-cp27mu-linux_x86_64.whl
  ARG CRED="server:123server123"
  RUN  curl -u ${CRED} ftp://yifileserver/DOCKER_IMAGES/Tensorflow/CPU/${TFLOW} -o ${TFLOW} && \
       pip --no-cache-dir install --upgrade /${TFLOW} && \
       rm -f /${TFLOW}   


################# Setting UP Environment ###################

ENV CI_BUILD_PYTHON=python \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    PYTHON_BIN_PATH=/usr/bin/python \
    PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages
	
############################################################
    
    
################ INTEL MKL SUPPORT #################

ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ARG CRED="server:123server123"
RUN cd /usr/local/lib && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libiomp5.so -o libiomp5.so && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libmklml_gnu.so -o libmklml_gnu.so && \
    curl -u ${CRED} ftp://yifileserver/IT/YiIT/lib/libmklml_intel.so -o libmklml_intel.so
    
####################################################


#################################
# Check Tensorflow Installation #
#################################
 
COPY cpu_tf_check.py /


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
