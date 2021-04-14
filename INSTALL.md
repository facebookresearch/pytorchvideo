# Installation

## Installing PytorchVideo


### 1. Install from PyPI
For stable release,
```
pip install pytorchvideo
=======
conda create -n pytorchvideo python=3.7
conda activate pytorchvideo
conda install -c pytorch pytorch=1.8.0 torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore -c iopath fvcore=0.1.4 iopath
```

For nightly builds,
```
pip install pytorchvideo-nightly
```

### 2. Install from GitHub using pip
```
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
```
To install using the code of the released version instead of from the main branch, use the following instead.
```
pip install "git+https://github.com/facebookresearch/pytorchvideo.git@stable"
```

### 3. Install from a local clone
```
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .

# For developing and testing
pip install -e . [test,dev]
```


## Requirements

### Core library

- Python 3.7 or 3.8 
- PyTorch 1.8.0 or higher.
- torchvision that matches the PyTorch installation. You can install them together as explained at pytorch.org to make sure of this.
- [fvcore](https://github.com/facebookresearch/fvcore) version 0.1.4 or higher
- [ioPath](https://github.com/facebookresearch/iopath)
- If CUDA is to be used, use a version which is supported by the corresponding pytorch version and at least version 10.2 or higher.

We recommend setting up a conda environment with Pytorch and Torchvision before installing PyTorchVideo.
For instance, follow the bellow instructions to setup the conda environment,
```
conda create -n pytorchvideo python=3.7
conda activate pytorchvideo
conda install -c pytorch pytorch=1.8.0 torchvision cudatoolkit=10.2
```

## Testing

Please follow the instructions mentioned in [test-README](https://github.com/facebookresearch/pytorchvideo/blob/master/tests/README.md) to run the provided tests.

## Linting

We also provide a linting script to correctly format your code edits.
Please follow the instructions mentioned in [dev-README](https://github.com/facebookresearch/pytorchvideo/blob/master/dev/README.md) to run the linter.
