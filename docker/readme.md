# Plato Docker images
## Usage:
Docker images offer a convenient way to quickly get started using Plato.  A convenient workflow is:
1. Create the problem definition in the host filesystem.  Users can clone/download the [Plato Engine](https://github.com/platoengine/platoengine/tree/docker) repository which has a collection of example problems.  More experienced users may be interested in the [Use Cases](https://github.com/platoengine/use_cases) repository.
2. Start the Plato container following the instructions below and run the optimization problem(s) of interest.  The container mounts the problem directory so results are available in the host filesystem.
3. Exit the container (or just open another terminal) to visualize the results. 

### Starting a Plato container
Images are available at [hub.docker.com](https://hub.docker.com/u/plato3d) for the 'release' and 'develop' branches:  
- plato3d/plato-analyze:cpu-develop
- plato3d/plato-analyze:cpu-release
- plato3d/plato-analyze:cuda-10.2-cc-7.5-develop
- plato3d/plato-analyze:cuda-10.2-cc-7.5-release

**Run as root (not recommended):** These can be used without modification, but the default user within the image is root.  This creates a few issues:
1. Any files created in mounted directories will have root ownership.
2. Running MPI programs (e.g., Plato) as root requires the user to set environment variables to permit it.
3. Running MPI programs as root will also induce warnings to stdout during runtime.

To run the 'root' image:
```shell
sudo docker run \
-v $(pwd):/examples \
--env OMPI_ALLOW_RUN_AS_ROOT=1 \
--env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
-it plato3d/plato-analyze:cpu-release
```
The command above sets two environment variables that are required to execute mpirun as root.  The -v argument followed by $(pwd):examples mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /examples.

**Run as user (recommended):** The images described above can be customized to an individual user.  This avoids the issues with running as root within the container.  To do so:
```shell
sudo docker build . \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g) \
-f analyze/Dockerfile.adduser \
-t plato-analyze:cpu-release-user
```

To run the resulting docker image:
```shell
sudo docker run \
-v $(pwd):/home/user/mount \
-it plato-analyze:cpu-release-user
```

The -v argument followed by $(pwd):/home/user/mount mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /home/user/mount.
