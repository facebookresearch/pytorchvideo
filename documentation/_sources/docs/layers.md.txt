# Overview


PyTorchVideo is an open source video understanding library that provides up to date builders for state of the art video understanding backbones, layers, heads, and losses addressing different tasks, including acoustic event detection, action recognition (video classification), action detection (video detection), multimodal understanding (acoustic visual classification), self-supervised learning.

The layers subpackage contains definitions for the following layers and activations:


* Layer
    * [BatchNorm](https://arxiv.org/abs/1502.03167)
    * [2+1 Conv](https://arxiv.org/abs/1711.11248)
    * ConCat
    * MLP
    * [Nonlocal Net](https://arxiv.org/abs/1711.07971)
    * Positional Encoding
    * [Squeeze and Excitation](https://arxiv.org/abs/1709.01507)
    * [Swish](https://arxiv.org/abs/1710.05941)

## Build standard models

PyTorchVideo provide default builders to construct state-of-the-art video understanding layers and activations.


### Layers

You can construct a layer with random weights by calling its constructor:

```
import pytorchvideo.layers as layers

nonlocal = layers.create_nonlocal(dim_in=256, dim_inner=128)
swish = layers.Swish()
conv_2plus1d = layers.create_conv_2plus1d(in_channels=256, out_channels=512)
```

You can verify whether you have built the model successfully by:

```
import pytorchvideo.layers as layers

nonlocal = layers.create_nonlocal(dim_in=256, dim_inner=128)
B, C, T, H, W = 2, 256, 4, 14, 14
input_tensor = torch.zeros(B, C, T, H, W)
output = nonlocal(input_tensor)

swish = layers.Swish()
B, C, T, H, W = 2, 256, 4, 14, 14
input_tensor = torch.zeros(B, C, T, H, W)
output = swish(input_tensor)

conv_2plus1d = layers.create_conv_2plus1d(in_channels=256, out_channels=512)
B, C, T, H, W = 2, 256, 4, 14, 14
input_tensor = torch.zeros(B, C, T, H, W)
output = conv_2plus1d(input_tensor)
```
