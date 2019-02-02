FROM ubuntu:16.04
MAINTAINER caffe-maint@googlegroups.com

#ENV http_proxy proxy:port
#ENV https_proxy proxy:port

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	bzip2 \
        build-essential \
        cmake \
        git \
        wget \
        ssh \
        openssh-server \
        numactl \
        vim \
        net-tools \
        iputils-ping \
        ufw \
        iptables && \
    rm -rf /var/lib/apt/lists/*

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

RUN mkdir /var/run/sshd && \
    echo 'root:intelcaffe@123' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/;s/Port 22/Port 10010/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 10010
RUN ssh-keygen -t rsa -A && \
    mkdir ~/.ssh && \
    touch ~/.ssh/config && \
    echo "Host *" > ~/.ssh/config && \
    echo "Port 10010" >> ~/.ssh/config && \
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
CMD ["/usr/sbin/sshd", "-D"]
