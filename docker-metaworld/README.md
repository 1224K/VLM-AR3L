# Docker image for MetaWorld

A quick reference for using MetaWorld in docker container. (ref RL-VLM-F)

## Build the Docker Images

```sh
docker build -t 1224k/metaworld .
```

## Run the Docker Container

```sh
# stop and remove the container if exists
docker stop metaworld && docker rm metaworld
# launch the container
docker run --name metaworld -it --gpus all -p 8080:8080 -d -v $PWD:/workspace 1224k/metaworld tail -f /dev/null
# exec into the container
docker exec -it metaworld /bin/bash
```

