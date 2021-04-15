---
id: tutorial_accelerator_use_model_transmuter
title: Accelerate your model with model transmuter in PytorchVideo/Accelerator
---


## Introduction
Got your own model, but still want to fully leverage efficient blocks in PytorchVideo/Accelerator? No problem, model transmuter can help you.
Model transmuter is a utility in PytorchVideo/Accelerator that takes user defined model, and replace modules in user model with equivalent efficient block when possible.
In this tutorial, we will go through typical steps of using model transmuter, including:
- Use model transmuter to replace modules in user model with efficient blocks
- Convert model into deploy form and deploy

## Use model transmuter to replace modules in user model with efficient blocks
First, let's assume user has following model to be transmuted:


```python
import torch
import torch.nn as nn

class user_model_residual_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem0 = nn.Conv3d(3, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.stem1 = nn.Conv3d(3, 3, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.pw = nn.Conv3d(3, 6, kernel_size=1)
        self.relu = nn.ReLU()
        self.dw = nn.Conv3d(6, 6, kernel_size=3, padding=1, groups=6)
        self.relu1 = nn.ReLU()
        self.pwl = nn.Conv3d(6, 3, kernel_size=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.stem0(x)
        out = self.stem1(out)
        out = self.pw(out)
        out = self.relu(out)
        out = self.dw(out)
        out = self.relu1(out)
        out = self.pwl(out)
        return self.relu2(out + x)
```

Then, let's use model transmuter by importing transmuter for targeting device. In this tutorial, we are using mobile cpu as example. Therefore we will import (1) model transmuter for mobile cpu and (2) top-level wrapper of model transmuter.


```python
import pytorchvideo.accelerator.deployment.mobile_cpu.transmuter  # mobile cpu model transmuter
from pytorchvideo.accelerator.deployment.common.model_transmuter import transmute_model  # top-level wrapper of model transmuter
```

We instantiate one user_model_residual_block, and transmute it by calling `transmute_model` with argument of `target_device="mobile_cpu"`


```python
model_transmute = user_model_residual_block()
transmute_model(
    model_transmute,
    target_device="mobile_cpu",
)
```

If we print the model, we will find that the some of modules in model has been replaced. In geenral, model transmuter will replace one submodule if its equivalent efficient block is found, otherwise that submodule will be kept intact.


## Convert model into deploy form and deploy
Now the model is ready to deploy. First of all, let's convert the model into deploy form. In order to do that, we need to use `convert_to_deployable_form` utility and provide an example input tensor to the model. `convert_to_deployable_form` will convert any instance of `EfficientBlockBase` (base class for efficient blocks in PytorchVideo/Accelerator) into deploy form, while leave other modules unchanged.
Note that once the model is converted into deploy form, the input size should be the same as the example input tensor size during conversion.


```python
# Define example input tensor
input_blob_size = (1, 3, 4, 6, 6)
input_tensor = torch.randn(input_blob_size)
```


```python
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
model_transmute_deploy = convert_to_deployable_form(
    model_transmute, input_tensor
)
```

Currently model transmuter only supports fp32 operation, and it will support int8 with incoming torch.fx quantization mode. In this tutorial, we assume deploy transmuted model without quantization. In this case, all we need to do is to export jit trace and then apply `optimize_for_mobile` for final optimization.


```python
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
)
traced_model = torch.jit.trace(model_transmute_deploy, input_tensor, strict=False)
traced_model_opt = optimize_for_mobile(traced_model)
# Here we can save the traced_model_opt to JIT file using traced_model_opt.save(<file_path>)
```
