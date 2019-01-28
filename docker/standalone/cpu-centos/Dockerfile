FROM centos:7
MAINTAINER caffe-maint@googlegroups.com

#ENV http_proxy proxy:port
#ENV https_proxy proxy:port

RUN rpm -iUvh https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm && \
    yum upgrade -y && \
    yum install -y \
        bzip2 \
        numactl \
        redhat-rpm-config \
        tar \
        findutils \
        gcc-c++ \
        cmake \
        git \
        vim \
        wget \
        ssh \
        openssh.x86_64 \
        openssh-server.x86_64 \
        openssh-clients.x86_64 \
        initscripts  \
        net-tools \
        ufw \
        iptables \
        gcc-gfortran && \
    yum clean all

# Install conda and Intel Caffe conda package
WORKDIR /root/
RUN wget --no-check-certificate -c https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \ 
    bash Miniconda2-latest-Linux-x86_64.sh -b && \ 
    ./miniconda2/bin/conda config --add channels intel && \ 
    ./miniconda2/bin/conda install -c intel caffe && \ 
    rm -rf /root/miniconda2/pkgs/* && \
    rm ~/Miniconda2-latest-Linux-x86_64.sh -f && \
    echo "export PATH=/root/miniconda2/bin:$PATH" >> /root/.bashrc
WORKDIR /root/miniconda2/caffe

# If you want to deploy container in HOST network mode, Install SSH service and config it to Non-standard Port. Or you needn’t below rows.
RUN mkdir /var/run/sshd && \
    echo "root:intelcaffe@123" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/;s/#Port 22/Port 10086/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 10086
RUN ssh-keygen -t rsa -A && \
    mkdir ~/.ssh && \
    touch ~/.ssh/config && \
    echo "Host *" > ~/.ssh/config && \
    echo "Port 10086" >> ~/.ssh/config && \
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
CMD ["/usr/sbin/sshd", "-D"]
