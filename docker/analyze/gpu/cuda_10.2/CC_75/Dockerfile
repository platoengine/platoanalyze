FROM plato3d/plato-spack:cuda-10.2

RUN spack install platoanalyze+cuda+mpmd+python+geometry+esp @develop ^nvcc-wrapper compute_capability=75 ^openmpi@4.0.2
RUN echo 'spack load platoanalyze' >> /etc/profile.d/spack.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
