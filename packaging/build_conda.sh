#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

mkdir -p packaging/out

version=$(python -c "exec(open('pytorchvideo/__init__.py').read()); print(__version__)")
export BUILD_VERSION=$version

conda build -c pytorch \
            -c defaults \
            -c conda-forge \
            -c iopath \
            -c fvcore \
            --no-anaconda-upload --python \
            "$PYTHON_VERSION" --output-folder packaging/out packaging/pytorchvideo