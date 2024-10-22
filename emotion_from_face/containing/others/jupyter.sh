#!/usr/bin/env bash
docker run \
  --interactive \
  --tty \
  --rm \
  --publish 1221:1221 \
  jupyter/datascience-notebook:latest start.sh jupyter lab/ml-workspace:latest
