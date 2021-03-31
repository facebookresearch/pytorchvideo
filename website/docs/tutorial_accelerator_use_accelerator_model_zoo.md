---
id: tutorial_accelerator_use_accelerator_model_zoo
title: Use PytorchVideo/Accelerator Model Zoo
---


## Introduction
This tutorial goes through how to use model zoo provided by PytorchVideo/Accelerator. To use model zoo in PytorchVideo/Accelerator, we should generally follow several steps:
- Use model builder to build selected model; 
- Load pretrain checkpoint;
- (Optional) Finetune;
- Deploy.

## Use model builder to build selected model
We use model builder in PytorchVideo/Accelerator model zoo to build pre-defined efficient model. Here we use EfficientX3D-XS (for mobile_cpu) as an example. For more available models and details, please refer to [this page].

EfficientX3D-XS is an implementation of X3D-XS network as described in [X3D paper](https://arxiv.org/abs/2004.04730) using efficient blocks. It is arithmetically equivalent with X3D-XS, but our benchmark on mobile phone shows 4.6X latency reduction compared with vanilla implementation.

In order to build EfficientX3D-XS, we simply do the following:


```python
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import EfficientX3d
model_efficient_x3d_xs = EfficientX3d(expansion='XS', head_act='identity')
```

Note that now the efficient blocks in the model are in original form, so the model is good for further training.

## Load pretrain checkpoint and (optional) finetune
For each model in model zoo, we provide pretrain checkpoint state_dict for model in original form. See [this page] for details about checkpoints and where to download them.


```python
from torch.hub import load_state_dict_from_url
checkpoint_path = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth'
checkpoint = load_state_dict_from_url(checkpoint_path)

model_efficient_x3d_xs.load_state_dict(checkpoint)
```

Now the model is ready for fine-tune. 

## Deploy
Now the model is ready to deploy. First of all, let's convert the model into deploy form. In order to do that, we need to use `convert_to_deployable_form` utility and provide an example input tensor to the model. Note that once the model is converted into deploy form, the input size should be the same as the example input tensor size during conversion.


```python
import torch
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
input_blob_size = (1, 3, 4, 160, 160)
input_tensor = torch.randn(input_blob_size)
model_efficient_x3d_xs_deploy = convert_to_deployable_form(model_efficient_x3d_xs, input_tensor)
```

Next we have two options: either deploy floating point model, or quantize model into int8 and then deploy.

Let's first assume we want to deploy floating point model. In this case, all we need to do is to export jit trace and then apply `optimize_for_mobile` for final optimization.


```python
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
)
traced_model = torch.jit.trace(model_efficient_x3d_xs_deploy, input_tensor, strict=False)
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
model_efficient_x3d_xs_deploy_quant_stub_wrapper = quant_stub_wrapper(model_efficient_x3d_xs_deploy)
```

Preparation step of quantization. Fusion has been done for efficient blocks automatically during `convert_to_deployable_form`, so we can just proceed to `torch.quantization.prepare`


```python
model_efficient_x3d_xs_deploy_quant_stub_wrapper.qconfig = torch.quantization.default_qconfig
model_efficient_x3d_xs_deploy_quant_stub_wrapper_prepared = torch.quantization.prepare(model_efficient_x3d_xs_deploy_quant_stub_wrapper)
```

Calibration and quantization. After preparation we will do calibration of quantization by feeding calibration dataset (skipped here) and then do quantization.


```python
# calibration is skipped here.
model_efficient_x3d_xs_deploy_quant_stub_wrapper_quantized = torch.quantization.convert(model_efficient_x3d_xs_deploy_quant_stub_wrapper_prepared)
```

Then we can export trace of int8 model and deploy on mobile devices.


```python
traced_model_int8 = torch.jit.trace(model_efficient_x3d_xs_deploy_quant_stub_wrapper_quantized, input_tensor, strict=False)
traced_model_int8_opt = optimize_for_mobile(traced_model_int8)
# Here we can save the traced_model_opt to JIT file using traced_model_int8_opt.save(<file_path>)
```

