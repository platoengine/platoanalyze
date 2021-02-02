FROM plato3d/plato-spack:cpu

RUN spack install platoanalyze~cuda~amgx+mpmd build_type=Debug @develop ^openmpi@4.0.2 ^trilinos build_type=Debug ^omega-h build_type=Debug ^platoengine build_type=Debug
RUN echo 'spack load platoanalyze' >> /etc/profile.d/spack.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
