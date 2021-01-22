# Dockerfiles
- Dockerfile.root generates an image based on plato3d/plato-spack:cuda-10.2 that includes plato analyze compiled for Nvidia compute capability of 7.5.  The user in the container is root.  Note that calling mpirun as root requires setting environment variables to explicitly allow it.
- Dockerfile.nonroot is the same as Dockerfile.root except the user in the container is 'plato'.  This avoids any issues with calling mpirun as root, but requires an additional step when mounting host filespace to make the mounted volume accessible.

## Building
To build the Dockerfile.root image:

```shell
sudo docker build . -f Dockerfile.root -t plato3d/plato-analyze:cuda-10.2-cc-7.5-release
```

To build the Dockerfile.nonroot image:

```shell
sudo docker build . -f Dockerfile.nonroot -t plato3d/plato-analyze:cuda-10.2-cc-7.5-nonroot-release
```

To build the develop branch, or any other branch, edit the Dockerfile and change @release to @develop, etc.

## Commiting
To commit the image to docker hub:

```shell
sudo docker push plato3d/plato-analyze:cuda-10.2-cc-7.5-release
sudo docker push plato3d/plato-analyze:cuda-10.2-cc-7.5-release-nonroot
```

## Using
To run the 'as root' docker image:

```shell
sudo docker run -v $(pwd):/examples --env OMPI_ALLOW_RUN_AS_ROOT=1 --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 --gpus all -it plato3d/plato-analyze:cuda-10.2-cc-7.5-release
```

The command above sets two environment variables that are required to execute mpirun as root.  The -v argument followed by $(pwd):examples mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /examples.

Be aware that running MPI programs (e.g., Plato) within a container will produce warning messages in the console that look like the following:

```shell
[8deb1e1269d0:00065] Read -1, expected 32784, errno = 1
```

These are benign and can be ignored.  To prevent these warnings add '--privileged' to the docker run arguments.
