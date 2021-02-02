## 
FROM plato3d/plato-base:cpu
MAINTAINER Joshua Robbins <jrobbin@sandia.gov>

##### Install platoengine/spack ###################################################################################
##
###################################################################################################################

ENV SPACK_ROOT=/usr/local

# load spack environment on login
RUN echo "source $SPACK_ROOT/share/spack/setup-env.sh" > /etc/profile.d/spack.sh

# install spack
RUN curl -s -L https://api.github.com/repos/platoengine/spack/tarball | tar xzC $SPACK_ROOT --strip 1

# install software
RUN spack install tar && spack clean -a
