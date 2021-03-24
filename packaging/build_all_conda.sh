#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

mkdir -p packaging/out

version=$(python -c "exec(open('pytorchvideo/__init__.py').read()); print(__version__)")
export BUILD_VERSION=$version

for PYTORCH_VERSION in 1.8.0
do
    for PYTHON_VERSION in 3.7 3.8 3.9
    do
        export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
        export PYTORCH_VERSION_NODOT=${PYTORCH_VERSION//./}
        conda build -c pytorchvideo_fair -c pytorch -c defaults -c conda-forge -c fvcore -c iopath -c anaconda --no-anaconda-upload --python "$PYTHON_VERSION" --output-folder packaging/out packaging/pytorchvideo
    done
done

ls -Rl packaging

for dir in  linux-64
do
    this_out_dir=packaging/output_files/$dir
    mkdir -p $this_out_dir
    cp packaging/out/$dir/*.tar.bz2 $this_out_dir
done

ls -Rl packaging
