#!/usr/bin/env bash
echo "creating container"
docker container create  \
  --publish 1234:1234 \
  --name notebook_container \
  --memory 3g \
  --interactive \
   notebook_image 
