


## Model Zoo and Benchmarks

PyTorchVideo provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup. All the models can be downloaded from the provided links.

### Kinetics-400

arch     | depth | pretrain | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model
-------- | ----- | -------- | -------------------------- | ----- | ----- | ----------------- | ---------- | --------------------------------------------------------------------------------------------------
C2D      | R50   | \-       | 8x8                        | 71.46 | 89.68 | 25.89 x 3 x 10    | 24.33      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D\_8x8\_R50.pyth)
I3D      | R50   | \-       | 8x8                        | 73.27 | 90.70 | 37.53 x 3 x 10    | 28.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D\_8x8\_R50.pyth)
Slow     | R50   | \-       | 4x16                       | 72.40 | 90.18 | 27.55 x 3 x 10    | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW\_4x16\_R50.pyth)
Slow     | R50   | \-       | 8x8                        | 74.58 | 91.63 | 54.52 x 3 x 10    | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW\_8x8\_R50.pyth)
SlowFast | R50   | \-       | 4x16                       | 75.34 | 91.89 | 36.69 x 3 x 10    | 34.48      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_4x16\_R50.pyth)
SlowFast | R50   | \-       | 8x8                        | 76.94 | 92.69 | 65.71 x 3 x 10    | 34.57      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_8x8\_R50.pyth)
SlowFast | R101  | \-       | 8x8                        | 77.90 | 93.27 | 127.20 x 3 x 10   | 62.83      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_8x8\_R101.pyth)
CSN      | R101  | \-       | 32x2                       | 77.00 | 92.90 | 75.62 x 3 x 10    | 22.21      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN\_32x2\_R101.pyth)
R(2+1)D  | R50   | \-       | 16x4                       | 76.01 | 92.23 | 76.45 x 3 x 10    | 28.11      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D\_16x4\_R50.pyth)
X3D      | XS    | \-       | 4x12                       | 69.12 | 88.63 | 0.91 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_XS.pyth)
X3D      | S     | \-       | 13x6                       | 73.33 | 91.27 | 2.96 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_S.pyth)
X3D      | M     | \-       | 16x5                       | 75.94 | 92.72 | 6.72 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_M.pyth)

### Something-Something V2

| arch     | depth | pretrain     | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model |
| -------- | ----- | ------------ | -------------------------- | ----- | ----- | ----------------- | ---------- | ----- |
| Slow     | R50   | Kinetics 400 | 8x8                        | 60.04 | 85.19 | 55.10 x 3 x 1     | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOW\_8x8\_R50.pyth)  |
| SlowFast | R50   | Kinetics 400 | 8x8                        | 61.68 | 86.92 | 66.60 x 3 x 1     | 34.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOWFAST\_8x8\_R50.pyth)   |


### Charades

| arch     | depth | pretrain     | frame length x sample rate | MAP   | Flops (G) x views | Params (M) | Model |
| -------- | ----- | ------------ | -------------------------- | ----- | ----------------- | ---------- | ----- |
| Slow     | R50   | Kinetics 400 | 8x8                        | 34.72 | 55.10 x 3 x 10    | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/charades/SLOW\_8x8\_R50.pyth)  |
| SlowFast | R50   | Kinetics 400 | 8x8                        | 37.24 | 66.60 x 3 x 10    | 34.00      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/charades/SLOWFAST\_8x8\_R50.pyth)   |

### Using PyTorchVideo model zoo
We provide several different ways to use PyTorchVideo model zoo.
* The models have been integrated into TorchHub, so could be loaded with TorchHub with or without pre-trained models. Additionally, we provide a [tutorial](https://pytorchvideo.org/docs/tutorial_torchhub_inference) which goes over the steps needed to load models from TorchHub and perform inference.
* PyTorchVideo models/datasets are also supported in PySlowFast. You can use [PySlowFast workflow](https://github.com/facebookresearch/SlowFast/tree/master/projects/pytorchvideo) to train or test PyTorchVideo models/datasets.
* You can also use [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to build training/test pipeline for PyTorchVideo models and datasets. Please check this [tutorial](https://pytorchvideo.org/docs/tutorial_classification) for more information.


Notes:
* The above benchmarks are conducted by [PySlowFast workflow](https://github.com/facebookresearch/SlowFast/tree/master/projects/pytorchvideo) using PyTorchVideo datasets and models.
* For more details on the data preparation, you can refer to [PyTorchVideo Data Preparation](data_preparation.md).
* For `Flops x views` column, we report the inference cost with a single “view" × the number of views (FLOPs × space_views × time_views). For example, we take 3 spatial crops for 10 temporal clips on Kinetics.



### PytorchVideo Accelerator Model Zoo
Accelerator model zoo provides a set of efficient models on target device with pretrained checkpoints. To learn more about how to build model, load checkpoint and deploy, please refer to [Use PyTorchVideo/Accelerator Model Zoo](https://pytorchvideo.org/docs/tutorial_accelerator_use_accelerator_model_zoo).

**Efficient Models for mobile CPU**
All top1/top5 accuracies are measured with 10-clip evaluation. Latency is benchmarked on Samsung S8 phone with 1s input clip length.

| model  | model builder                                                            | top 1 | top 5 | latency (ms) | params (M) | checkpoint          |
|--------|--------------------------------------------------------------------------|-------|-------|--------------|----------------|---------------------|
| X3D_XS | models. accelerator. mobile_cpu. efficient_x3d. EfficientX3d (expansion="XS") | 68.5  | 88.0  |          233 | 3.8            | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth) |
| X3D_S  | models. accelerator. mobile_cpu. efficient_x3d. EfficientX3d (expansion="S")  | 73.0  | 90.6  |          764 | 3.8            | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_s_original_form.pyth) |
