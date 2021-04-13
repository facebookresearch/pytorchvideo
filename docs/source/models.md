# Overview


PyTorchVideo is an open source video understanding library that provides up to date builders for state of the art video understanding backbones, layers, heads, and losses addressing different tasks, including acoustic event detection, action recognition (video classification), action detection (video detection), multimodal understanding (acoustic visual classification), self-supervised learning.

The models subpackage contains definitions for the following model architectures and layers:


* Acoustic Backbone
    * Acoustic ResNet
* Visual Backbone
    * [I3D](https://arxiv.org/pdf/1705.07750.pdf)
    * [C2D](https://arxiv.org/pdf/1711.07971.pdf)
    * [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
    * [Nonlocal Networks](https://arxiv.org/pdf/1711.07971.pdf)
    * [R2+1D](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)
    * CSN
    * [SlowFast](https://arxiv.org/pdf/1812.03982.pdf)
    * [X3D](https://arxiv.org/pdf/2004.04730.pdf)
* Self-Supervised Learning
    * [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
    * [Bootstrap Your Own Latent](https://arxiv.org/pdf/2006.07733.pdf)
    * [Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)


## Build standard models

PyTorchVideo provide default builders to construct state-of-the-art video understanding models, layers, heads, and losses. 

### Models

You can construct a model with random weights by calling its constructor:

```
import pytorchvideo.models as models

resnet = models.create_resnet()
acoustic_resnet = models.create_acoustic_resnet()
slowfast = models.create_slowfast()
x3d = models.create_x3d()
r2plus1d = models.create_r2plus1d()
csn = models.create_csn()
```

You can verify whether you have built the model successfully by:

```
import pytorchvideo.models as models

resnet = models.create_resnet()
B, C, T, H, W = 2, 3, 8, 224, 224
input_tensor = torch.zeros(B, C, T, H, W)
output = resnet(input_tensor)
```

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

### Heads

You can construct a head with random weights by calling its constructor:

```
import pytorchvideo.models as models

res_head = models.head.create_res_basic_head(in_features, out_features)
x3d_head = models.x3d.create_x3d_head(dim_in=1024, dim_inner=512, dim_out=2048, num_classes=400)
```

You can verify whether you have built the head successfully by:

```
import pytorchvideo.models as models

res_head = models.head.create_res_basic_head(in_features, out_features)
B, C, T, H, W = 2, 256, 4, 14, 14
input_tensor = torch.zeros(B, C, T, H, W)
output = res_head(input_tensor)

x3d_head = models.x3d.create_x3d_head(dim_in=1024, dim_inner=512, dim_out=2048, num_classes=400)
B, C, T, H, W = 2, 256, 4, 14, 14
input_tensor = torch.zeros(B, C, T, H, W)
output = x3d_head(input_tensor)
```

### Losses

You can construct a loss by calling its constructor:

```
import pytorchvideo.models as models

simclr_loss = models.SimCLR()
```

You can verify whether you have built the loss successfully by:

```
import pytorchvideo.models as models
import pytorchvideo.layers as layers

resnet = models.create_resnet()
mlp = layers.make_multilayer_perceptron(fully_connected_dims=(2048, 1024, 2048))
simclr_loss = models.SimCLR(mlp=mlp, backbone=resnet)
B, C, T, H, W = 2, 256, 4, 14, 14
view1, view2 = torch.zeros(B, C, T, H, W), torch.zeros(B, C, T, H, W)
loss = simclr_loss(view1, view2)
```

## Build customized models

PyTorchVideo also supports building models with customized components, which is an important feature for video understanding research. Here we take a standard stem model as an example, show how to build each resnet components (head, backbone, stem) separately, and how to use your customized components to replace standard components.


```
from pytorchvideo.models.stem import create_res_basic_stem


# Create standard stem layer.
stem = create_res_basic_stem(in_channels=3, out_channels=64)

# Create customized stem layer with YourFancyNorm
stem = create_res_basic_stem(
    in_channels=3, 
    out_channels=64, 
    norm=YourFancyNorm,  # GhostNorm for example
)

# Create customized stem layer with YourFancyConv
stem = create_res_basic_stem(
    in_channels=3, 
    out_channels=64, 
    conv=YourFancyConv,  # OctConv for example
)

# Create customized stem layer with YourFancyAct
stem = create_res_basic_stem(
    in_channels=3, 
    out_channels=64, 
    activation=YourFancyAct,  # Swish for example
)

# Create customized stem layer with YourFancyPool
stem = create_res_basic_stem(
    in_channels=3, 
    out_channels=64, 
    pool=YourFancyPool,  # MinPool for example
)

```
