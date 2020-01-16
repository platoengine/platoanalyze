# Plato Analyze
Fast physics with gradients for multidisciplinary analysis and optimization


## Building [Plato Analyze](https://github.com/platoengine/platoanalyze)

The recommended way to build and install Plato Analyze is with the [Plato fork](https://github.com/platoengine/spack) of [Spack](https://spack.io).

```shell
git clone https://github.com/platoengine/spack.git
source spack/share/spack/setup-env.sh
spack install platoanalyze ^nvcc-wrapper compute_capability=$COMPUTE_CAPABILITY
```

Where $COMPUTE_CAPABILITY is the compute capability of your GPU. For example, for an nVidia Tesla V100 GPU, you would run

```shell
spack install platoanalyze ^nvcc-wrapper compute_capability=70
```

For more information on building and configuring Plato Analyze, please see the [wiki](https://github.com/platoengine/platoengine/wiki)
