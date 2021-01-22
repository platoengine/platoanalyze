# Dockerfiles
- Dockerfile.root generates an image based on plato3d/plato-base:cuda-10.2 that includes platoengine/spack.  The user in the container is root.  Note that calling mpirun as root requires setting environment variables to explicitly allow it.
- Dockerfile.nonroot is the same as Dockerfile.root except the user in the container is 'plato'.  This avoids any issues with calling mpirun as root, but requires an additional step when mounting host filespace to make the mounted volume accessible.

## Building
To build the Dockerfile.root image:

```shell
sudo docker build . -f Dockerfile.root -t plato3d/plato-spack:cuda-10.2
```

To build the Dockerfile.nonroot image:

```shell
sudo docker build . -f Dockerfile.nonroot -t plato3d/plato-spack:cuda-10.2-nonroot
```

## Commiting
To commit the image to docker hub:
```shell
sudo docker push plato3d/plato-spack:cuda-10.2
sudo docker push plato3d/plato-spack:cuda-10.2-nonroot
```
