#!/bin/bash

set -e

export IMAGE=gcr.io/chai-959f8/training:ppo-trainer-ziyi
docker build --cache-from $IMAGE -t $IMAGE .
docker push $IMAGE
