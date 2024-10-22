#!/usr/bin/env bash
bash "./container_stop.sh"
bash "./container_remove.sh"
bash "./image_remove.sh"
docker system prune
# docker system prune --all # last resort
