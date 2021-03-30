---
id: doc_model_zoo
title: PyTorchVideo Model Zoo
sidebar_label: Model zoo and benchmarks
---

## PyTorchVideo Model Zoo and Benchmarks

PyTorchVideo provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup. All the models can be downloaded from the provided links.

### Kinetics-400

arch     | depth | pretrain | frame length x sample rate | top 1 | top 5 | Flops (G) | Params (M) | Model                                                                                             
-------- | ----- | -------- | -------------------------- | ----- | ----- | --------- | ---------- | --------------------------------------------------------------------------------------------------
C2D      | R50   | \-       | 8x8                        | 71.46 | 89.68 | 25.89     | 24.33      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D\_8x8\_R50.pyth)      
I3D      | R50   | \-       | 8x8                        | 73.27 | 90.70 | 37.53     | 28.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D\_8x8\_R50.pyth)      
Slow     | R50   | \-       | 4x16                       | 72.40 | 90.18 | 27.55     | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW\_4x16\_R50.pyth)    
Slow     | R50   | \-       | 8x8                        | 74.58 | 91.63 | 54.52     | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW\_8x8\_R50.pyth)     
SlowFast | R50   | \-       | 4x16                       | 75.34 | 91.89 | 36.69     | 34.48      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_4x16\_R50.pyth)
SlowFast | R50   | \-       | 8x8                        | 76.94 | 92.69 | 65.71     | 34.57      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_8x8\_R50.pyth) 
SlowFast | R101  | \-       | 8x8                        | 77.90 | 93.27 | 127.20    | 62.83      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_8x8\_R101.pyth)
CSN      | R101  | \-       | 32x2                       | 77.00 | 92.90 | 75.62     | 22.21      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN\_32x2\_R101.pyth)    
R(2+1)D  | R50   | \-       | 16x4                       | 76.01 | 92.23 | 76.45     | 28.11      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D\_16x4\_R50.pyth)
X3D      | XS    | \-       | 4x12                       | 69.12 | 88.63 | 0.91      | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_XS.pyth)            
X3D      | S     | \-       | 13x6                       | 73.33 | 91.27 | 2.96      | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_S.pyth)             
X3D      | M     | \-       | 16x5                       | 75.94 | 92.72 | 6.72      | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_M.pyth)    

### Something-Something V2
| arch     | depth | pretrain     | frame length x sample rate | top 1 | top 5 | Flops (G) | Params (M) | Model |
| -------- | ----- | ------------ | -------------------------- | ----- | ----- | --------- | ---------- | ----- |
| Slow     | R50   | Kinetics 400 | 8x8                        | 60.04 | 85.19 | 55.10     | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOW\_8x8\_R50.pyth)  |
| SlowFast | R50   | Kinetics 400 | 8x8                        | 61.68 | 86.92 | 66.60     | 34.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOWFAST\_8x8\_R50.pyth)   |


### Charades
| arch     | depth | pretrain     | frame length x sample rate | MAP   | Flops (G) | Params (M) | Model |
| -------- | ----- | ------------ | ---------------- | ----- | --------- | ---------- | ----- |
| Slow     | R50   | Kinetics 400 | 8x8              | 34.72 | 55.10     | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/charades/SLOW\_8x8\_R50.pyth)  |
| SlowFast | R50   | Kinetics 400 | 8x8              | 37.24 | 66.60     | 34.00      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/charades/SLOWFAST\_8x8\_R50.pyth)   |


Notes:
* The above benchmarks are conducted by [PySlowFast workflow]() using PyTorchVideo datasets and models.
* For more details on the data preparation, you can refer to [PyTorchVideo Data Preparation](doc_model_zoo_data).


## Use PytorchVideo model zoo
We provide several different ways to use PyTorchVideo model zoo.
* The models have been integrated into TorchHub, so could be loaded with TorchHub with or without pre-trained models. Additionally, we provide a [tutorial]() which goes over the steps needed to load models from TorchHub and perform inference.
* PyTorchVideo models are also supported in PySlowFast. You can use [PySlowFast workflow]() to train or test PyTorchVideo models. 