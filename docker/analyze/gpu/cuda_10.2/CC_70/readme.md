# Dockerfiles
- Dockerfile generates an image based on plato3d/plato-spack:cuda-10.2 that includes plato analyze compiled for Nvidia compute capability of 7.0.  The user in the container is root.  Note that calling mpirun as root requires setting environment variables to explicitly allow it.

## Building
To build the image:

```shell
sudo docker build . --no-cache -f Dockerfile -t plato3d/plato-analyze:cuda-10.2-cc-7.0-develop
```

To build the release branch, or any other branch, edit the Dockerfile and change @develop to @release, etc.

## Commiting
To commit the image to docker hub:

```shell
sudo docker push plato3d/plato-analyze:cuda-10.2-cc-7.0-develop
```

## Using
To run the docker image:

```shell
sudo docker run -v $(pwd):/examples --env OMPI_ALLOW_RUN_AS_ROOT=1 --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 --gpus all -it plato3d/plato-analyze:cuda-10.2-cc-7.0-develop
```

The command above sets two environment variables that are required to execute mpirun as root.  The -v argument followed by $(pwd):examples mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /examples.

Be aware that running MPI programs (e.g., Plato) within a container will produce warning messages in the console that look like the following:

```shell
[8deb1e1269d0:00065] Read -1, expected 32784, errno = 1
```

These are benign and can be ignored.  To prevent these warnings add '--privileged' to the docker run arguments.
