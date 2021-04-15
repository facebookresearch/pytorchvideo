---
id: tutorial_accelerator_build_your_model
title: Build your efficient model with PytorchVideo/Accelerator
---


## Introduction
In this tutorial, we will go through:
- Basics of efficient blocks in PytorchVideo/Accelerator;
- Design, train and deploy a model composed of efficient blocks for mobile CPU.

## Basics of efficient blocks in PytorchVideo/Accelerator
Efficient blocks are blocks with high efficiency. For a target device, we benchmark efficiency of basic network components and provide a collection of efficient blocks under `pytorchvideo/layers/accelerator/<target_device>` (for simple layers) and `pytorchvideo/models/accelerator/<target_device>` (for complex modules such as residual block). Inferencing of a model built up with corresponding efficient blocks on target device is guranteed to be efficient.

Each efficient block module is an instance of nn.Module, and has two forms: **original form** (for training) and **deploy form** (for inference). When in original form, the efficient block module has exactly the same behavior as a corresponding vanilla nn.Module for both forward and backward operation. User can freely mix and match efficient blocks for the same target device and build up their own model. Once model is built and trained, user can convert each efficient block in model into deploy form. The conversion will do graph and kernel optimization on each efficient block, and efficient block in deploy form is arithmetically equivalent to original form but has much higher efficiency during inference. 

## Design, train and deploy a model composed of efficient blocks for mobile CPU
### Build a model
In this section, let's go through the process of design, train and deploy using a example toy model using efficient blocks under `pytorchvideo/layers/accelerator/mobile_cpu` and `pytorchvideo/models/accelerator/mobile_cpu`, which includes:
- One conv3d head layer with 5x1x1 kernel followed by ReLU activation;
- One residual block with squeeze-excite;
- One average pool and fully connected layer as final output.

First, let's import efficient blocks.


```python
# Imports
import torch.nn as nn
from pytorchvideo.layers.accelerator.mobile_cpu.activation_functions import (
    supported_act_functions,
)
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d5x1x1BnAct,
)
from pytorchvideo.models.accelerator.mobile_cpu.residual_blocks import (
    X3dBottleneckBlock,
)
from pytorchvideo.layers.accelerator.mobile_cpu.pool import AdaptiveAvgPool3dOutSize1
from pytorchvideo.layers.accelerator.mobile_cpu.fully_connected import FullyConnected

```

Then we can build a model using those efficient blocks.


```python
class MyNet(nn.Module):
    def __init__(
        self,
        in_channel=3,  # input channel of first 5x1x1 layer
        residual_block_channel=24,  # input channel of residual block
        expansion_ratio=3, # expansion ratio of residual block
        num_classes=4, # final output classes
    ):
        super().__init__()
        # s1 - 5x1x1 conv3d layer
        self.s1 = Conv3d5x1x1BnAct(
            in_channel,
            residual_block_channel,
            bias=False,
            groups=1,
            use_bn=False,
        )
        # s2 - residual block
        mid_channel = int(residual_block_channel * expansion_ratio)
        self.s2 = X3dBottleneckBlock(
                in_channels=residual_block_channel,
                mid_channels=mid_channel,
                out_channels=residual_block_channel,
                use_residual=True,
                spatial_stride=1,
                se_ratio=0.0625,
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
        # Average pool and fully connected layer
        self.avg_pool = AdaptiveAvgPool3dOutSize1()
        self.projection = FullyConnected(residual_block_channel, num_classes, bias=True)
        self.act = supported_act_functions['relu']()

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        # Performs fully convolutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)

        return x
```

We can instantiate MyNet and its efficient blocks will be in original form.


```python
net_inst = MyNet()
print(net_inst)
```

    MyNet(
      (s1): Conv3d5x1x1BnAct(
        (kernel): Sequential(
          (conv): Conv3d(3, 24, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), bias=False)
          (act): ReLU(
            (act): ReLU(inplace=True)
          )
        )
      )
      (s2): X3dBottleneckBlock(
        (_residual_add_func): FloatFunctional(
          (activation_post_process): Identity()
        )
        (final_act): ReLU(
          (act): ReLU(inplace=True)
        )
        (layers): Sequential(
          (conv_0): Conv3dPwBnAct(
            (kernel): Sequential(
              (conv): Conv3d(24, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
              (bn): BatchNorm3d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): ReLU(
                (act): ReLU(inplace=True)
              )
            )
          )
          (conv_1): Conv3d3x3x3DwBnAct(
            (kernel): Sequential(
              (conv): Conv3d(72, 72, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=72, bias=False)
              (bn): BatchNorm3d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity(
                (act): Identity()
              )
            )
          )
          (se): SqueezeExcitation(
            (se): SqueezeExcitation(
              (block): Sequential(
                (0): Conv3d(72, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                (1): ReLU()
                (2): Conv3d(8, 72, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                (3): Sigmoid()
              )
            )
          )
          (act_func_1): Swish(
            (act): Swish()
          )
          (conv_2): Conv3dPwBnAct(
            (kernel): Sequential(
              (conv): Conv3d(72, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
              (bn): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity(
                (act): Identity()
              )
            )
          )
        )
      )
      (avg_pool): AdaptiveAvgPool3dOutSize1(
        (pool): AdaptiveAvgPool3d(output_size=1)
      )
      (projection): FullyConnected(
        (model): Linear(in_features=24, out_features=4, bias=True)
      )
      (act): ReLU(
        (act): ReLU(inplace=True)
      )
    )


### Train model
Then we can train the model with your dataset/optimizer. Here we skip this training step, and just leave the weight as initial value.

### Deploy model
Now the model is ready to deploy. First of all, let's convert the model into deploy form. In order to do that, we need to use `convert_to_deployable_form` utility and provide an example input tensor to the model. Note that once the model is converted into deploy form, the input size should be the same as the example input tensor size during conversion.


```python
import torch
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
input_blob_size = (1, 3, 4, 6, 6)
input_tensor = torch.randn(input_blob_size)
net_inst_deploy = convert_to_deployable_form(net_inst, input_tensor)

```

We can see that the network graph has been changed after conversion, which did kernel and graph optimization.


```python
print(net_inst_deploy)
```

    MyNet(
      (s1): Conv3d5x1x1BnAct(
        (kernel): Sequential(
          (conv): _Conv3dTemporalKernel5Decomposed(
            (_conv2d_0): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (_conv2d_1): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (_conv2d_2): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (_conv2d_3): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (_conv2d_4): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (_add_funcs): ModuleList(
              (0): FloatFunctional(
                (activation_post_process): Identity()
              )
              (1): FloatFunctional(
                (activation_post_process): Identity()
              )
              (2): FloatFunctional(
                (activation_post_process): Identity()
              )
              (3): FloatFunctional(
                (activation_post_process): Identity()
              )
              (4): FloatFunctional(
                (activation_post_process): Identity()
              )
              (5): FloatFunctional(
                (activation_post_process): Identity()
              )
              (6): FloatFunctional(
                (activation_post_process): Identity()
              )
              (7): FloatFunctional(
                (activation_post_process): Identity()
              )
              (8): FloatFunctional(
                (activation_post_process): Identity()
              )
              (9): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (_cat_func): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (act): ReLU(
            (act): ReLU(inplace=True)
          )
        )
      )
      (s2): X3dBottleneckBlock(
        (_residual_add_func): FloatFunctional(
          (activation_post_process): Identity()
        )
        (final_act): ReLU(
          (act): ReLU(inplace=True)
        )
        (layers): Sequential(
          (conv_0): Conv3dPwBnAct(
            (kernel): Sequential(
              (0): _Reshape()
              (1): Sequential(
                (conv): ConvReLU2d(
                  (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
                  (1): ReLU(inplace=True)
                )
                (bn): Identity()
                (act): ReLU(
                  (act): Identity()
                )
              )
              (2): _Reshape()
            )
          )
          (conv_1): Conv3d3x3x3DwBnAct(
            (kernel): Sequential(
              (conv): _Conv3dTemporalKernel3Decomposed(
                (_conv2d_3_3_0): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
                (_conv2d_3_3_2): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
                (_conv2d_3_3_1): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72)
                (_add_funcs): ModuleList(
                  (0): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                  (1): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                  (2): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                  (3): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                  (4): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                  (5): FloatFunctional(
                    (activation_post_process): Identity()
                  )
                )
                (_cat_func): FloatFunctional(
                  (activation_post_process): Identity()
                )
              )
              (bn): Identity()
              (act): Identity(
                (act): Identity()
              )
            )
          )
          (se): SqueezeExcitation(
            (se): _SkipConnectMul(
              (layer): Sequential(
                (0): AdaptiveAvgPool3d(output_size=1)
                (1): _Reshape()
                (2): Linear(in_features=72, out_features=8, bias=True)
                (3): ReLU()
                (4): Linear(in_features=8, out_features=72, bias=True)
                (5): Sigmoid()
                (6): _Reshape()
              )
              (mul_func): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
          )
          (act_func_1): Swish(
            (act): _NaiveSwish(
              (mul_func): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
          )
          (conv_2): Conv3dPwBnAct(
            (kernel): Sequential(
              (0): _Reshape()
              (1): Sequential(
                (conv): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
                (bn): Identity()
                (act): Identity(
                  (act): Identity()
                )
              )
              (2): _Reshape()
            )
          )
        )
      )
      (avg_pool): AdaptiveAvgPool3dOutSize1(
        (pool): AvgPool3d(kernel_size=(4, 6, 6), stride=(4, 6, 6), padding=0)
      )
      (projection): FullyConnected(
        (model): Linear(in_features=24, out_features=4, bias=True)
      )
      (act): ReLU(
        (act): ReLU(inplace=True)
      )
    )


Let's check whether the network after conversion is arithmetically equivalent. We expect the output to be very close before/after conversion, with some small difference due to numeric noise from floating point operation.


```python
net_inst.eval()
out_ref = net_inst(input_tensor)
out = net_inst_deploy(input_tensor)

max_err = float(torch.max(torch.abs(out_ref - out)))
print(f"max error is {max_err}")
```

    max error is 2.9802322387695312e-08


Next we have two options: either deploy floating point model, or quantize model into int8 and then deploy.

Let's first assume we want to deploy floating point model. In this case, all we need to do is to export jit trace and then apply `optimize_for_mobile` for final optimization.


```python
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
)
traced_model = torch.jit.trace(net_inst_deploy, input_tensor, strict=False)
traced_model_opt = optimize_for_mobile(traced_model)
# Here we can save the traced_model_opt to JIT file using traced_model_opt.save(<file_path>)
```

Alternatively, we may also want to deploy a quantized model. Efficient blocks are quantization-friendly by design - just wrap the model in deploy form with `QuantStub/DeQuantStub` and it is ready for Pytorch eager mode quantization.


```python
# Wrapper class for adding QuantStub/DeQuantStub.
class quant_stub_wrapper(nn.Module):
    def __init__(self, module_in):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = module_in
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
```


```python
net_inst_quant_stub_wrapper = quant_stub_wrapper(net_inst_deploy)
```

Preparation step of quantization. Fusion has been done for efficient blocks automatically during `convert_to_deployable_form`, so we can just proceed to `torch.quantization.prepare`


```python
net_inst_quant_stub_wrapper.qconfig = torch.quantization.default_qconfig
net_inst_quant_stub_wrapper_prepared = torch.quantization.prepare(net_inst_quant_stub_wrapper)
```

Calibration and quantization. After preparation we will do calibration of quantization by feeding calibration dataset (skipped here) and then do quantization.


```python
# calibration is skipped here.
net_inst_quant_stub_wrapper_quantized = torch.quantization.convert(net_inst_quant_stub_wrapper_prepared)
```


Then we can export trace of int8 model and deploy on mobile devices.


```python
traced_model_int8 = torch.jit.trace(net_inst_quant_stub_wrapper_quantized, input_tensor, strict=False)
traced_model_int8_opt = optimize_for_mobile(traced_model_int8)
# Here we can save the traced_model_opt to JIT file using traced_model_int8_opt.save(<file_path>)
```

