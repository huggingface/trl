#!/bin/bash

set -e

IMAGE=gcr.io/chai-959f8/training:ppo-trainer-ziyi
echo "Building image '$IMAGE'"

SSH_PRIV_KEY=`cat ~/.ssh/id_rsa`

docker build -t "$IMAGE"  --build-arg ssh_priv_key="$SSH_PRIV_KEY" .
docker push "$IMAGE"
