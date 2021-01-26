## 
FROM       ubuntu:18.04
MAINTAINER Joshua Robbins <jrobbin@sandia.gov>

# general environment for docker
ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_UNSAFE_CONFIGURE=1

# install minimal dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       autoconf \
       build-essential \
       ca-certificates \
       coreutils \
       curl \
       file \
       tclsh \
       environment-modules \
       git \
       gfortran \
       python \
       unzip \
       vim \
       ssh \
       time \
       gdb \
       python3-distutils

## add git-lfs just in case:
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs


##### Install cuda base ###########################################################################################
##
## excerpted from:
## https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.2/ubuntu18.04-x86_64/base/Dockerfile
##
###################################################################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"


##### Install cuda runtime ########################################################################################
##
## excerpted from:
### https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.2/ubuntu18.04-x86_64/runtime/Dockerfile
##
###################################################################################################################

ENV NCCL_VERSION 2.7.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-npp-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    libcublas10=10.2.2.89-1 \
    libnccl2=$NCCL_VERSION-1+cuda10.2 \
    && apt-mark hold libnccl2 \
    && rm -rf /var/lib/apt/lists/*


##### Install cuda devel ##########################################################################################
##
## excerpted from:
### https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.2/ubuntu18.04-x86_64/devel/Dockerfile
##
###################################################################################################################

ENV NCCL_VERSION 2.7.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvml-dev-$CUDA_PKG_VERSION \
    cuda-command-line-tools-$CUDA_PKG_VERSION \
    cuda-nvprof-$CUDA_PKG_VERSION \
    cuda-npp-dev-$CUDA_PKG_VERSION \
    cuda-libraries-dev-$CUDA_PKG_VERSION \
    cuda-minimal-build-$CUDA_PKG_VERSION \
    libcublas-dev=10.2.2.89-1 \
    libnccl-dev=2.7.8-1+cuda10.2 \
    && apt-mark hold libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
