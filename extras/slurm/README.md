## Training

This file contains instructions how to run the training on a Nvidia-DGX with SLURM.

<br/>

Start a training:

```bash
sbatch Scribosermo/extras/slurm/run_training.sh
```

View training output

```bash
tail -f slurm-JOBID.out
```

Stop a training:

```bash
scancel JOBID
```

<br/>

#### Use docker/podman images:

- Install singularity on your local pc: [Instructions](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps)

- Convert your images to compressed singularity images:

  ```bash
  # docker
  ./Scribosermo/extras/slurm/convert_to_sifgz.sh d scribosermo
  # podman
  ./Scribosermo/extras/slurm/convert_to_sifgz.sh p scribosermo
  ```

- Upload and decompress the singularity image

  ```bash
  scp scribosermo.sif.gz user@ip:~/images/
  gzip -d scribosermo.sif.gz
  ```

<br/>

## Debugging

Test gpu availability (edit run_training.sh and execute the following command):

```bash
python3 -c "import tensorflow as tf; sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))"
```

Convert image.sif to sandbox folder (if local /tmp directory is to small). In commands simply exchange image.sif with sandbox directory:

```bash
singularity build --sandbox dsgs_sandbox/ scribosermo.sif
```

<br/>

## Installation of dependencies

Installation of singularity and needed dependencies without using root privileges. \
File structure will look as follows:

```
my_speech2text_folder (named db_xds here)
    checkpoints
    data_original
    data_prepared
    Scribosermo            <- This repositiory

    programs <- New folder for dependencies installations
```

<br/>

Install _go_ language:

```bash
wget https://dl.google.com/go/go1.14.linux-amd64.tar.gz
tar -xzf go1.14.linux-amd64.tar.gz
rm go1.14.linux-amd64.tar.gz

export PATH=$PATH:/cfs/share/cache/db_xds/programs/go/bin
```

Install _openssl_:

```bash
wget https://www.openssl.org/source/openssl-1.1.1d.tar.gz
tar -xzf openssl-1.1.1d.tar.gz
rm openssl-1.1.1d.tar.gz

cd openssl-1.1.1d
./config --prefix=/cfs/share/cache/db_xds/programs/openssl-1.1.1d/build --openssldir=/cfs/share/cache/db_xds/programs/openssl-1.1.1d/build -Wl,-rpath=/cfs/share/cache/db_xds/programs/openssl-1.1.1d/build/lib  # --openssldir and -Wl,-rpath needed?
make && make install

export PATH=$PATH:/cfs/share/cache/db_xds/programs/openssl-1.1.1d/build/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cfs/share/cache/db_xds/programs/openssl-1.1.1d/build/lib
```

Install _singularity_:

```bash
export VERSION=3.5.3  # Adjust version as necessary
wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz

tar -xzf singularity-${VERSION}.tar.gz
rm singularity-${VERSION}.tar.gz
cd singularity


./mconfig --without-suid --prefix=/cfs/share/cache/db_xds/programs/singularity  # Adjust path as necessary

# If linking 'libssl' and 'libuuid' does not work, edit 'mlocal/checks/project-post.chk' and comment out the checks for those two libraries. For me building did work then.

make -C ./builddir
make -C ./builddir install

export PATH=$PATH:/cfs/share/cache/db_xds/programs/singularity/builddir
# Export permanently
echo "export PATH=$PATH:/cfs/share/cache/db_xds/programs/singularity/builddir" >> ~/.bashrc

# Install spython for dockerfile conversion
# (but building does not work without root privileges)
pip3 install spython
# Close and reopen shell
```

Test singularity:

```bash
singularity exec \
  --nv \
  --bind checkpoints/:/checkpoints/ \
  --bind data_original/:/data_original/ \
  --bind data_prepared/:/data_prepared/ \
  --bind Scribosermo/:/Scribosermo/ \
  scribosermo.sif
```

<br/>

Download from google drive:

```bash
pip3 install gdown

# Get file id of files in a directory by clicking on it and then "more-options->open in new window"
# Then copy file id from url and insert it below
gdown https://drive.google.com/uc?id=
```
