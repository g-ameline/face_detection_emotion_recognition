#!/usr/bin/env bash
echo "starting container"
docker container start \
  --interactive \
  --attach \
   notebook_container

