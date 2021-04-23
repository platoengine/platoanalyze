# Dockerfile
This Dockerfile generates an image based on Ubuntu 18.04 that includes all of the base requirements for spack

## Building
To build the image:

```shell
sudo docker build . -f Dockerfile -t plato3d/plato-base:cpu
```

## Commiting
To commit the image to docker hub:
```shell
sudo docker push plato3d/plato-base:cpu
```
