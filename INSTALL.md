# Installation


## Requirements

### Core library

The core library is written in PyTorch with few components leveraged from TorchVision and OpenCV-Python. We recommend using GPU environments
for the best performace.

- Linux
- Python 3.7 or 3.8 
- PyTorch 1.8.0 or higher.
- torchvision that matches the PyTorch installation. You can install them together as explained at pytorch.org to make sure of this.
- [fvcore](https://github.com/facebookresearch/fvcore) version 0.1.4 or higher
- [ioPath](https://github.com/facebookresearch/iopath)
- If CUDA is to be used, use a version which is supported by the corresponding pytorch version and at least version 10.2 or higher.

The runtime dependencies can be installed by running:
```
conda create -n pytorchvideo python=3.7
conda activate pytorchvideo
conda install -c pytorch pytorch=1.8.0 torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore -c iopath fvcore=0.1.4 iopath
pip install opencv-python 
```

### Tests/Linting and Demos

For developing on top of PyTorchVideo or contributing, you will need to run the linter and tests. If you want to run any of the notebook tutorials or examples you will need to install the additional dependencies.

For the precise versions of these additional dependecies, we recommend looking at `setup.py` in the root of the project.

- black
- isort
- flake8
- autoflake
- jupyter
- pytest
- coverage


## Installing PytorchVideo
After installing the above dependencies, run one of the following commands:

### 1. Install from Anaconda Cloud, on Linux only
The builds are updated **nightly**,
```
# Only to be run after installing requirements
conda install -c pytorchvideo pytorchvideo
```

### 2. Install from GitHub using pip
```
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
```
To install using the code of the released version instead of from the main branch, use the following instead.
```
pip install "git+https://github.com/facebookresearch/pytorchvideo.git@stable"
```

### 3. Install from PyPI
The wheels are updated only on **stable releases**,
```
pip install pytorchvideo

# For developing and testing
pip install pytorchvideo[test,dev]
```

### 4. Install from a local clone
```
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo 
pip install -e .

# For developing and testing
pip install -e . [test,dev]
```

