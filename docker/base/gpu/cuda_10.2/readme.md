# Dockerfile
This Dockerfile generates an image based on Ubuntu 18.04 that includes all of the base requirements for spack as well as the Cuda 10.2 base, runtime, and devel libraries

## Building
To build the image:

```shell
sudo docker build . -f Dockerfile -t plato3d/plato-base:cuda-10.2
```

## Commiting
To commit the image to docker hub:
```shell
sudo docker push plato3d/plato-base:cuda-10.2
```
