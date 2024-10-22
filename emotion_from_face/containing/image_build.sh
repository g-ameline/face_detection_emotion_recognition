#!/usr/bin/env bash
virtual_environment_filepath=$(< virtual_environment_filepath.txt)
echo "filepath to vitual environment :"
echo "  $virtual_environment_filepath"

kernel_name=$(< kernel_name.txt)
echo "kernel name :"
echo "  $kernel_name"

port=$(< port.txt)
echo "port :"
echo "  $port"

echo " building image"
docker image build  \
  --progress plain \
  --no-cache \
  --file ./Dockerfile \
  --build-arg VIRTUAL_ENVIRONMENT_FILEPATH=$virtual_environment_filepath \
  --build-arg KERNEL_NAME=$kernel_name \
  --build-arg PORT=$port \
  --pull \
  --tag notebook_image \
  ../ 
