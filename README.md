<p align="center">
 <img width="130%" src="./.github/logo_horizontal_color.svg" />
</p>

<p align="center">
   <a href="https://github.com/facebookresearch/pytorchvideo/blob/oss_website_docs/LICENSE">
    <img src="https://img.shields.io/pypi/l/pytorchvideo" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/pytorchvideo/">
    <img src="https://img.shields.io/pypi/v/pytorchvideo?color=blue&label=release" alt="CircleCI" />
  </a>
    <a href="https://circleci.com/gh/facebookresearch/pytorchvideo/tree/oss_website_docs">
    <img src="https://img.shields.io/circleci/build/github/facebookresearch/pytorchvideo/oss_website_docs?token=efdf3ff5b6f6acf44f4af39b683dea31d40e5901" alt="Coverage" />
  </a>
    <a href="https://codecov.io/gh/facebookresearch/pytorchvideo/branch/oss_website_docs">
    <img src="https://codecov.io/gh/facebookresearch/pytorchvideo/branch/oss_website_docs/graph/badge.svg?token=OSZSI6JU31"/>
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

Get started with PyTorchVideo by trying out one of our [tutorials](https://pytorchvideo.org/tutorials/) or by running examples in the [projects folder](https://github.com/facebookresearch/pytorchvideo/tree/master/projects).


## Model Zoo and Baselines
We provide a large set of baseline results and trained models available for download in the [PyTorchVideo Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/master/docs/source/model_zoo.md).

## Contributors

PyTorchVideo is written and maintained by the Facebook AI Research.

## Development

We welcome new contributions to PyTorchVideo and we will be actively maintaining this library! Please refer to [`CONTRIBUTING.md`](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

<!-- ## Citation

If you find PyTorchVideo useful in your research or wish to refer to the baseline results published in the [Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/master/docs/source/model_zoo.md), please use the following BibTeX entry.

```bibtex
@Misc{pytorchvideo2019,
  author =       {Facebook AI Research},
  title =        {PytorchVideo - A library for accelerating video research},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/facebookresearch/pytorchvideo}
}
``` -->
