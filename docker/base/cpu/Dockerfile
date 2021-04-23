## 
FROM       ubuntu:18.04
MAINTAINER Joshua Robbins <jrobbin@sandia.gov>

# general environment for docker
ENV DEBIAN_FRONTEND=noninteractive \
    NSAFE_CONFIGURE=1              \
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

