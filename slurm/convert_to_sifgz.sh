#! /bin/bash

if [[ "$1" == "d" ]]; then
    echo "Saving docker image as tar file ..."
    docker save -o tmp_dimg.tar $2

elif [[ "$1" == "p" ]]; then
    echo "Saving podman image as tar file ..."
    podman save -o tmp_dimg.tar --format oci-archive $2

else
    echo "Image type not matching"
    exit
fi

echo "Converting image to singularity file ..."
mkdir /tmp/sing/
export SINGULARITY_LOCALCACHEDIR=/tmp/sing/
export SINGULARITY_CACHEDIR=/tmp/sing/
export SINGULARITY_TMPDIR=/tmp/sing/
if [[ "$1" == "d" ]]; then
    singularity build -d "$2".sif docker-archive://tmp_dimg.tar
else
    singularity build -d "$2".sif oci-archive://tmp_dimg.tar
fi

echo "Compressing singularity image ..."
gzip -9 -f "$2".sif

echo "Deleting tmp files ..."
rm -rf /tmp/sing/
rm tmp_dimg.tar
