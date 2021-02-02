# Dockerfiles
- Dockerfile.adduser generates an image based on one of the plato-analyze images.  The new image includes a user named 'user' that has the same id and group id as the user on the host.  This permits using the image without being root within the image.  

## Building
To build the user specific image:

```shell
sudo docker build . --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile.adduser -t plato-analyze:cpu-release-user
```

## Using
To run the resulting docker image:

```shell
sudo docker run -v $(pwd):/home/user/mount -it plato-analyze:cpu-release-user
```

The -v argument followed by $(pwd):/home/user/mount mounts the present working directory on the host (i.e., the result of 'pwd') inside the container at /home/user/mount.
