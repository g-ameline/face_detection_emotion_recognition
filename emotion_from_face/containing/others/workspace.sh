#!/usr/bin/env bash
docker run \
  --interactive \
  --tty \
  --rm \
  --publish 4321:4321 \
  mltooling/ml-workspace:latest
