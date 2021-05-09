<p align="center">
 <img width="130%" src="./.github/logo_horizontal_color.svg" />
</p>

<p align="center">
   <a href="https://github.com/facebookresearch/pytorchvideo/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/pytorchvideo" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/pytorchvideo/">
    <img src="https://img.shields.io/pypi/v/pytorchvideo?color=blue&label=release" alt="CircleCI" />
  </a>
    <a href="https://circleci.com/gh/facebookresearch/pytorchvideo/tree/master">
    <img src="https://img.shields.io/circleci/build/github/facebookresearch/pytorchvideo/master?token=efdf3ff5b6f6acf44f4af39b683dea31d40e5901" alt="Coverage" />
  </a>
    <a href="https://codecov.io/gh/facebookresearch/pytorchvideo/branch/master">
    <img src="https://codecov.io/gh/facebookresearch/pytorchvideo/branch/master/graph/badge.svg?token=OSZSI6JU31"/>
  </a>
  <p align="center">
    <i> A deep learning library for video understanding research.</i>
  </p>
  <p align="center">
    <i>Check the <a href="https://pytorchvideo.org/">website</a> for more information.</i>
  </p>
 </p>
 
## Introduction

PyTorchVideo is a deeplearning library with a focus on video understanding work. PytorchVideo provides resusable, modular and efficient components needed to accelerate the video understanding research. PyTorchVideo is developed using [PyTorch](https://pytorch.org) and supports different deeplearning video components like video models, video datasets, and video-specific transforms.

Key features include:

- **Based on PyTorch:** Built using PyTorch. Makes it easy to use all of the PyTorch-ecosystem components. 
- **Reproducible Model Zoo:** Variety of state of the art pretrained video models and their associated benchmarks that are ready to use.
  Complementing the model zoo, PyTorchVideo comes with extensive data loaders supporting different datasets.
- **Efficient Video Components:** Video-focused fast and efficient components that are easy to use. Supports accelerated inference on hardware.


## Installation

Install PyTorchVideo inside a conda environment(Python >=3.7) with
```shell
pip install pytorchvideo
```

For detailed instructions please refer to [INSTALL.md](INSTALL.md).

## License

PyTorchVideo is released under the [Apache 2.0 License](LICENSE).

## Tutorials

Get started with PyTorchVideo by trying out one of our [tutorials](https://pytorchvideo.org/docs/tutorial_overview) or by running examples in the [tutorials folder](./tutorials).


## Model Zoo and Baselines
We provide a large set of baseline results and trained models available for download in the [PyTorchVideo Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/master/docs/source/model_zoo.md).

## Contributors

Here is the growing list of PyTorchVideo contributors in alphabetical order (let us know if you would like to be added):
[Aaron Adcock](https://www.linkedin.com/in/aaron-adcock-79855383/), [Amy Bearman](https://www.linkedin.com/in/amy-bearman/), [Bernard Nguyen](https://www.linkedin.com/in/mrbernardnguyen/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Chengyuan Yan](https://www.linkedin.com/in/chengyuan-yan-4a804282/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Dave Schnizlein](https://www.linkedin.com/in/david-schnizlein-96020136/), [Haoqi Fan](https://haoqifan.github.io/), [Heng Wang](https://hengcv.github.io/), [Jackson Hamburger](https://www.linkedin.com/in/jackson-hamburger-986a2873/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/), [Kalyan Vasudev Alwala](https://www.linkedin.com/in/kalyan-vasudev-alwala-2a802b64/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/), [Nikhila Ravi](https://www.linkedin.com/in/nikhilaravi/), [Ross Girshick](https://www.rossgirshick.info/), [Tullie Murrell](https://www.linkedin.com/in/tullie/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Weiyao Wang](https://www.linkedin.com/in/weiyaowang/?locale=en_US), [Yanghao Li](https://lyttonhao.github.io/), [Yilei Li](https://liyilui.github.io/personal_page/), [Zhengxing Chen](http://czxttkl.github.io/), [Zhicheng Yan](https://www.linkedin.com/in/zhichengyan/).


## Development

We welcome new contributions to PyTorchVideo and we will be actively maintaining this library! Please refer to [`CONTRIBUTING.md`](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.
