---
id: tutorial_classification
title: Training a PyTorchVideo classification model
---

# Introduction

In this tutorial we will show how to build a simple video classification training pipeline using PyTorchVideo models, datasets and transforms. We'll be using a 3D ResNet [1] for the model, Kinetics [2] for the dataset and a standard video transform augmentation recipe. As PyTorchVideo doesn't contain training code, we'll use [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch training framework - to help out. Don't worry if you don't have Lightning experience, we'll explain what's needed as we go along.

[1] He, Kaiming, et al. Deep Residual Learning for Image Recognition. ArXiv:1512.03385, 2015.

[2] W. Kay, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

# Dataset

To start off with, let's setup the PyTorchVideo Kinetics data loader using a [pytorch_lightning_LightningDataModule](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.datamodule.html#pytorch_lightning.core.datamodule.LightningDataModule) . A LightningDataModule is a wrapper that defines the train, val and test data partitions, we'll use it to wrap the PyTorchVideo Kinetics dataset below.

The PyTorchVideo Kinetics dataset is just an alias for the general [pytorchvideo.data.EncodedVideoDataset](http://pytorchvideo.org/api/data/encoded_video.html#pytorchvideo.data.encoded_video_dataset.EncodedVideoDataset) class. If you look at its constructor, you'll notice that most args are what you'd expect (e.g. path to data). However, there are a few args that are more specific to PyTorchVideo datasets:
- video_sampler - defining the order to sample a video at each iteration. The default is a "random".
- clip_sampler - defining how to sample a clip from the chosen video at each iteration. For a train partition it is typical to use a "random" clip sampler (i.e. take a random clip of the specified duration from the video). For testing, typically you'll use "uniform" (i.e. uniformly sample all clips of the specified duration from the video) to ensure the entire video is sampled in each epoch.
- transform - this provides a way to apply user defined data preprocessing or augmentation before batch collating by the PyTorch data loader. We'll show an example using this later.


```python
import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  _DATA_PATH = <path_to_kinetics_data_dir>
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 8
  _NUM_WORKERS = 8  # Number of parallel processes fetching data

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train.csv
    """
    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "train.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/train.csv
    """
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "val.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )
```

# Transforms

As mentioned above, PyTorchVideo datasets take a "transform" callable arg that defines custom processing (e.g. augmentations, normalization) that's applied to each clip. The callable arg takes a clip dictionary defining the different modalities and metadata. pytorchvideo.data.Kinetics clips have the following dictionary format:

```python
  {
     'video': <video_tensor>,     # Shape: (C, T, H, W)
     'audio': <audio_tensor>,     # Shape: (S)
     'label': <action_label>,     # Integer defining class annotation
     'video_name': <video_path>,  # Video file path stem
     'video_index': <video_id>,   # index of video used by sampler
     'clip_index': <clip_id>      # index of clip sampled within video
  }
```

PyTorchVideo provides several transforms which you can see in the [docs](https://pytorchvideo.readthedocs.io/en/latest/transforms.html) Notably, PyTorchVideo provides dictionary transforms that can be used to easily interoperate with other domain specific libraries. For example, [pytorchvideo.transforms.ApplyTransformToKey(key, transform)](https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html), can be used to apply domain specific transforms to a specific dictionary key. For video tensors we use the same tensor shape as TorchVision and for audio we use TorchAudio tensor shapes, making it east to apply their transforms alongside PyTorchVideo ones.

Below we revise the LightningDataModule from the last section to include transforms coming from both TorchVision and PyTorchVideo. For brevity we'll just show the KineticsDataModule.train_dataloader method. The validation dataset transforms would be the same just without the augmentations (RandomShortSideScale, RandomCropVideo, RandomHorizontalFlipVideo).

```python
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip
)

class KineticsDataModule(pytorch_lightning.LightningDataModule):

# ...

    def train_dataloader(self):
      """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

# ...

```

# Model

All PyTorchVideo models and layers can be built with simple, reproducible factory functions. We call this the "flat" model interface because the args don't require hierachies of configs to be used. An example building a default ResNet can be found below. See the [docs](https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/models/resnet.html#create_bottleneck_block) for more configuration options.

```python
import pytorchvideo.models.resnet

def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=400, # Kinetics has 400 classes so we need out final head to align
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )
```

# Putting it all together

To put everything together, let's create a [pytorch_lightning.LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html). This defines the train and validation step code (i.e. the code inside the training and evaluation loops), and the optimizer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self):
      super().__init__()
      self.model = make_kinetics_resnet()

  def forward(self, x):
      return self.model(x)

  def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the 
      # format provided by the dataset
      y_hat = self.model(batch["video"])

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, batch["label"])

      # Log the train loss to Tensorboard
      self.log("train_loss", loss.item())

      return loss

  def validation_step(self, batch, batch_idx):
      y_hat = self.model(batch["video"])
      loss = F.cross_entropy(y_hat, batch["label"])
      self.log("val_loss", loss)
      return loss

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)
```

Our VideoClassificationLightningModule and KineticsDataModule are ready be trained together using the [pytorch_lightning.Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)!. The trainer class has many arguments to define the training environment (e.g. num_gpus, distributed_backend). To keep things simple we'll just use the default local cpu training but note that this would likely take weeks to train so you might want to use more performant settings based on your environment.

```python
  def train():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, data_module)
```

# Conclusion

In this tutorial we showed how to train a 3D ResNet on Kinetics using PyTorch Lightning. You can see the final code from the tutorial (including a few extra bells and whistles) in the PyTorchVideo projects directory.

To learn more about PyTorchVideo, check out the rest of the [documentation](https://pytorchvideo.readthedocs.io/en/latest/index.html)  and [tutorials](https://pytorchvideo.org/docs/tutorial_overview).

