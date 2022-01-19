## PyTorchVideo Trainer

A [PyTorch-Lightning]() based trainer supporting PytorchVideo models and dataloaders for various video understanding tasks.

Currently supported tasks include:

- Video Action Recognition: ResNet's, SlowFast Models, X3D models and MViT
- Video Self-Supervised Learning: SimCLR, BYOL, MoCo
- (Planned) Video Action Detection

## Installation

These instructions assumes that both pytorch and torchvision are already installed
using the instructions in [INSTALL.md](https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md#requirements)

Install the required additional dependency `recipes` by running the following command,
```
pip install "git+https://github.com/facebookresearch/recipes.git"
```

Post that, install PyTorchVideo Trainer by running,
```
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo/pytorchvideo_trainer
pip install -e .

# For developing and testing
pip install -e . [test,dev]
```

## Testing

Before running the tests, please ensure that you installed the necessary additional test dependencies.

Use the the following command to run the tests:
```
# From the current directory
python -m unittest discover -v -s ./tests
```
