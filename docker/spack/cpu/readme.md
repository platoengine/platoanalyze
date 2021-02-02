# Dockerfiles
- Dockerfile generates an image based on plato3d/plato-base:cpu that includes platoengine/spack.  The user in the container is root.  Note that calling mpirun as root requires setting environment variables to explicitly allow it.

## Building
To build the Dockerfile image:

```shell
sudo docker build . -f Dockerfile -t plato3d/plato-spack:cpu
```

## Commiting
To commit the image to docker hub:

```shell
sudo docker push plato3d/plato-spack:cpu
```
