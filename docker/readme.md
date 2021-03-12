# Plato Docker images
Docker images offer a convenient way to quickly get started using Plato.  
## Prerequisites:
1. [Docker](https://docs.docker.com/engine/install/) must be installed and running.
2. [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required if users wish to access GPU hardware in the Docker container.  Nvidia Docker is only available in Linux and Windows (using [Windows Subsystem Linux 2](https://docs.microsoft.com/en-us/windows/wsl/install-win10)).
## Usage:
A typical workflow is:
1. Create the problem definition in the host filesystem.  Users can clone/download the [Plato Engine](https://github.com/platoengine/platoengine/tree/docker) repository which has a collection of example problems.  More experienced users may be interested in the [Use Cases](https://github.com/platoengine/use_cases) repository.
2. Start the Plato container following the instructions below and run the optimization problem(s) of interest.  The container mounts the problem directory so results are available in the host filesystem.
3. Exit the container (or just open another terminal) to visualize the results. 

### Starting a Plato container
Images are available at [hub.docker.com](https://hub.docker.com/u/plato3d) for the 'release' and 'develop' branches:  
- plato3d/plato-analyze:cpu-develop
- plato3d/plato-analyze:cpu-release
- plato3d/plato-analyze:cuda-10.2-cc-7.5-develop
- plato3d/plato-analyze:cuda-10.2-cc-7.5-release

**Option 1 - Run as user (recommended):** The above images can be customized to an individual user.  This avoids issues with running as root within the container.  To do so, create a file called `Dockerfile.adduser` with the following contents:
```shell
FROM plato3d/plato-analyze:cpu-develop

ARG USER_ID
ARG GROUP_ID

USER root
RUN addgroup --gid $GROUP_ID user || true
RUN useradd --create-home --shell /bin/bash --uid $USER_ID --gid $GROUP_ID user
USER user
WORKDIR /home/user

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
```
where `FROM plato3d/plato-analyze:cpu-develop` indicates the image that you wish to customize. In this example, we'll customize the Plato image that's tagged `cpu-develop` to create a customized image tagged as `cpu-develop-user`. From the directory that contains `Dockerfile.adduser`, execute the following:
```shell
sudo docker build . --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile.adduser -t plato-analyze:cpu-develop-user
```

To use the resulting docker image, change to the directory that contains the problem(s) to be run and start the container:
- For CPU:
```shell
sudo docker run -v $(pwd):/home/user/mount --privileged -it plato-analyze:cpu-develop-user
```
- For GPU:
```shell
sudo docker run --gpus all -v $(pwd):/home/user/mount --privileged -it plato-analyze:cpu-develop-user
```

The -v argument followed by $(pwd):/home/user/mount mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /home/user/mount.

**Option 2 - Run as root (not recommended):** The Plato images can be used without modification, but the default user within the image is root.  This creates a few issues:
1. Any files created in mounted directories will be owned by root.
2. Running MPI programs (e.g., Plato) as root requires the user to set environment variables to permit it.
3. Running MPI programs as root will also induce warnings to stdout during runtime.

To run the 'root' image:
- For CPU:
```shell
sudo docker run -v $(pwd):/examples --env OMPI_ALLOW_RUN_AS_ROOT=1 --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -it plato3d/plato-analyze:cpu-release
```
- For GPU:
```shell
sudo docker run --gpus all -v $(pwd):/examples --env OMPI_ALLOW_RUN_AS_ROOT=1 --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -it plato3d/plato-analyze:cpu-release
```

The command above sets two environment variables that are required to execute mpirun as root.  The -v argument followed by $(pwd):examples mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /examples.
