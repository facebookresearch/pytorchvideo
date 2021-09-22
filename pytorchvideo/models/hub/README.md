## TorchHub Models

PyTorchVideo provides a large set of [TorchHub](https://pytorch.org/hub/) models for state-of-the-art models with pre-trained weights. Check the tables below for the torchhub names and corresponding models.


### Kinetics-400

Models are trained on Kinetics-400. For more benchmarking and model details, please check the [PyTorchVideo Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md)

torchhub name            | arch     | depth | frame length x sample rate | top 1 | top 5 |
------------------------ | -------- | ----- | -------------------------- | ----- | ----- |
c2d_r50                  | C2D      | R50   | 8x8                        | 71.46 | 89.68 |
i3d_r50                  | I3D      | R50   | 8x8                        | 73.27 | 90.70 |
slow_r50                 | Slow     | R50   | 8x8                        | 74.58 | 91.63 |
slowfast_r50             | SlowFast | R50   | 8x8                        | 76.94 | 92.69 |
slowfast_r101            | SlowFast | R101  | 8x8                        | 77.90 | 93.27 |
slowfast_16x8_r101_50_50 | SlowFast | R101  | 16x8                       | 78.70 | 93.61 |
csn_r101                 | CSN      | R101  | 32x2                       | 77.00 | 92.90 |
r2plus1d_r50             | R(2+1)D  | R50   | 16x4                       | 76.01 | 92.23 |
x3d_xs                   | X3D      | XS    | 4x12                       | 69.12 | 88.63 |
x3d_s                    | X3D      | S     | 13x6                       | 73.33 | 91.27 |
x3d_m                    | X3D      | M     | 16x5                       | 75.94 | 92.72 |
x3d_l                    | X3D      | L     | 16x5                       | 77.44 | 93.31 |

### PytorchVideo Accelerator Models

**Efficient Models for mobile CPU**
Models are trained on Kinetics-400. Latency is benchmarked on Samsung S8 phone with 1s input clip length.

torchhub name    | model  | top 1 | top 5 | latency (ms) |
---------------- |--------|-------|-------|--------------|
efficient_x3d_xs | X3D_XS | 68.5  | 88.0  |          233 |
efficient_x3d_s  | X3D_S  | 73.0  | 90.6  |          764 |



### Using PyTorchVideo torchhub models
The models have been integrated into TorchHub, so could be loaded with TorchHub with or without pre-trained models. You can specify the torchhub name for the model to construct the model with pre-trained weights:

```Python
# Pick a pretrained model
model_name = "slowfast_r50"
model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
```

Notes:
* Please check [torchhub inference tutorial](https://pytorchvideo.org/docs/tutorial_torchhub_inference) for more details about how to load models from TorchHub and perform inference.
* Check [Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md) for the full set of supported PytorchVideo model zoo and more details about how the model zoo is prepared.
