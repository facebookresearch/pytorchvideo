# Overview

The PyTorchVideo transforms package contains common video algorithms used for preprocessing and/or augmenting video data. The package also contains helper dictionary transforms that are useful for interoperability between PyTorchVideo datasets clip outputs (TODO link to sample datasets clip) and domain specific transforms. For example, here is a standard transform pipeline for a video model, that could be used with a PyTorchVideo dataset:

```python
transform = torchvision.transforms.Compose([
  pytorchvideo.transforms.ApplyTransformToKey(
    key="video",
    transform=torchvision.transforms.Compose([
      pytorchvideo.transforms.UniformTemporalSubsample(8),
      pytorchvideo.transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
      pytorchvideo.transforms.RandomShortSideScale(min_size=256, max_size=320),
      torchvision.transforms.RandomCrop(244),
      torchvision.transforms.RandomHorizontalFlip(p=0.5),
    )]
  )
])
dataset = pytorchvideo.data.Kinetics(
  data_path="path/to/kinetics_root/train.csv",
  clip_sampler=pytorchvideo.data.make_clip_sampler("random", duration=2),
  transform=transform
)
```

Notice how the example also includes transforms from TorchVision? PyTorchVideo uses the same canonical tensor shape as TorchVision for video and TorchAudio for audio. This allows the frameworks to be used together freely.

## Transform vs Functional interface

The example above demonstrated the pytorchvideo.transforms interface. These transforms are nn.module callable classes that can be stringed together in a declarative way. PyTorchVideo also provides a pytorchvideo.transforms.functional interface, which is essentially just the functions that the nn.module classes use. These allow more fine-grained control over the transformations and may be more suitable for use outside the dataset preprocessing use case.

## Scriptable transforms

All non-OpenCV transforms are TorchScriptable, as described in the [TorchVision docs](https://pytorch.org/vision/stable/transforms.html#scriptable-transforms), in order to script the transforms together, please use torch.nn.Sequential instead of torchvision.transform.Compose.
