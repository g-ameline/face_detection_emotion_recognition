#!/usr/bin/env bash
echo "running container"
docker container run\
  --publish 1234:1234
  --interactive \
  --tty \
  --name notebook_container
  --rm \
   notebook_container
docker run -it -p 1234:1234 notebook_image
