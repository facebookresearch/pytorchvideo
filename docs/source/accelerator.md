
# Overview

Our vision for PyTorchVideo/Accelerator is to enable video understanding models to run efficiently on all tiers of hardware devices, from mobile phone to GPU. PyTorchVideo/Accelerator (Accelerator) is aimed to accelerate the speed of video understanding model running on various hardware devices, as well as the whole process of design and deploy hardware-aware efficient video understanding models. Specifically, Accelerator provides a complete environment which allows users to:

* Design efficient models for target hardware with carefully tuned efficient blocks;
* Fine tune efficient model from Model Zoo;
* Optimize model kernel and graph for target device;
* Deploy efficient model to target device.


We benchmarked the latency of SOTA models ([X3D-XS and X3D-S](https://arxiv.org/abs/2004.04730)) on a mainstream mobile device (Samsung S9 International, released in 2018). With Accelerator, we not only observed 4-6X latency reduction on fp32, but also enabled int8 operation which has not been supported in vanilla Pytorch. A table summarizing latency comparison is shown below.

|model  |implementation |precision  |latency per 1-s clip (ms)  |speed up   |
|---    |-------------------------  |---    |---    |---    |
|X3D-XS |Vanilla Pytorch            |fp32   |1067   |1.0X   |
|X3D-XS |PytrochVideo/ Accelerator   |fp32   |233    |4.6X   |
|X3D-XS |PytrochVideo/ Accelerator   |int8   |165    |6.5X   |
|X3D-S  |Vanilla Pytorch            |fp32   |4248   |1.0X   |
|X3D-S  |PytrochVideo/ Accelerator   |fp32   |763    |5.6X   |
|X3D-S  |PytrochVideo/ Accelerator   |int8   |503    |8.4X   |

## Components in PyTorchVideo/Accelerator

### Efficient block library

Efficient block library contains common building blocks (residual block, squeeze-excite, etc.) that can be mapped to high-performance kernel operator implementation library of target device platform. The rationale behind having an efficient block library is that high-performance kernel operator library generally only supports a small set of kernel operators. In other words, a randomly picked kernel might not be supported by high-performance kernel operator library. By having an efficient block library and building model using efficient blocks in that library can guarantee the model is deployable with high efficiency on target device.

Efficient block library lives under `pytorchvideo/layers/accelerator/<target_device>` (for simple layers) and `pytorchvideo/models/accelerator/<target_device>` (for complex modules such as residual block). Please also check [Build your model with PyTorchVideo/Accelerator](https://pytorchvideo.org/docs/tutorial_accelerator_build_your_model) tutorial for detailed examples.

### Deployment

Deployment flow includes kernel optimization as well as model export for target backend. Kernel optimization utilities can be an extremely important part that decides performance of on-device model operation. Accelerator provides a bunch of useful utilities for deployment under `pytorchvideo/accelerator/deployment`. Please also check related tutorials ([Build your model with PyTorchVideo/Accelerator](https://pytorchvideo.org/docs/tutorial_accelerator_build_your_model), [Accelerate your model with model transmuter in PyTorchVideo/Accelerator](https://pytorchvideo.org/docs/tutorial_accelerator_use_model_transmuter))  for detailed examples.

### Model zoo

Accelerator provides efficient model zoo for target devices, which include model builder (under `pytorchvideo/models/accelerator/<target_device>`) as well as pretrained checkpoint. Please also refer to [Use PyTorchVideo/Accelerator Model Zoo](https://pytorchvideo.org/docs/tutorial_accelerator_use_accelerator_model_zoo) for how to use model zoo.


## Supported devices

Currently mobile cpu (ARM-based cpu on mobile phones) is supported. We will update this page once more target devices are supported.

## Demo

Checkout our on-device video classification demos running on mobile phone!

[Android demo](https://github.com/pytorch/android-demo-app/tree/master/TorchVideo)

[iOS demo](https://github.com/pytorch/ios-demo-app/tree/master/TorchVideo)

## Jumpstart

Refer to following tutorial pages to get started!

[Build your model with PyTorchVideo/Accelerator](https://pytorchvideo.org/docs/tutorial_accelerator_build_your_model)

[Use PyTorchVideo/Accelerator Model Zoo](https://pytorchvideo.org/docs/tutorial_accelerator_use_accelerator_model_zoo)

[Accelerate your model with model transmuter in PyTorchVideo/Accelerator](https://pytorchvideo.org/docs/tutorial_accelerator_use_model_transmuter)
