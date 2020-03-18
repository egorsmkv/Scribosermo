#! /bin/bash

echo "Saving docker image as tar file ..."
docker save -o tmp_dimg.tar $1

echo "Converting image to singularity file ..."
mkdir /tmp/sing/
export SINGULARITY_LOCALCACHEDIR=/tmp/sing/
export SINGULARITY_CACHEDIR=/tmp/sing/
export SINGULARITY_TMPDIR=/tmp/sing/
singularity build -d "$1".sif docker-archive://tmp_dimg.tar

echo "Compressing singularity image ..."
gzip -9 "$1".sif

echo "Deleting tmp files ..."
rm -rf /tmp/sing/
rm tmp_dimg.tar
