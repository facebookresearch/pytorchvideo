---
id: tutorial_classification_kinetics
title: Training a video classification model on Kinetics
---

# Introduction

In this tutorial we will show how to build a simple video classification training pipeline using PyTorchVideo models, datasets and transforms. We'll be using a 3D ResNet for the model, Kinetics for the dataset and a standard video transform augmentation recipe, however, any PyTorchVideo component should be able to be swaped in easily. As PyTorchVideo doesn't contain training code, we'll use PyTorchLightning to help out. Don't worry if you don't have PyTorchLightning experience, we'll explain what's needed as we go along.

# Dataset

To start off with, let's setup the pytorch_lightning_LightningDataModule to retrieve Kinetics video clips ready for training our model. The LightningDataModule defines the train, val and test dataloaders, as well as any data preparation or setup code.

The PyTorchVideo Kinetics dataset is an alias for the PyTorchVideo.data.EncodedVideoDataset class. You can see the class signature here (TODO link to sphinx docs). Other than standard args you'd notice in a PyTorch dataset class (e.g. path to data), PyTorchVideo datasets provide three new args:
- video_sampler - defining the order to sample a video at each iteration. The default is a RandomSampler.
- clip_sampler - defining how to sample a clip from the chosen video at each iteration. For a train partition it is typical to use a "random" clip sampler (i.e. take a random clip of the specified duration from the video). For testing, typically you'll use "uniform" (i.e. uniformly sample all clips of the specified duration from the video) to ensure the whole video is sampled.
- transform - this provides a way to apply user defined data preprocessing or augmentation before batch collating by the PyTorch DataLoader. We'll show an example using this later.

```
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
      Create the Kinetics train partition from the list of video labels in {self._DATA_PATH}/train.csv
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
      Create the Kinetics validation partition from the list of video labels in {self._DATA_PATH}/train.csv
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

As briefly mentioned above, PyTorchVideo datasets take a "transform" callable arg that defines any custom processing (e.g. augmentations, normalization) to be applied to each clip before it's returned. The callable arg takes a clip dictionary defining the different modalities and metadata. Kinetics clips have the following dictionary format:

```
  {
     'video': <video_tensor>,
     'audio': <audio_tensor>,
     'label': <action_label>,
     'video_name': <video_path>
     'video_index': <video_id>
     'clip_index': <clip_id>
  }
```

- "video" is a Tensor of shape (C, T, H, W)
- "audio" is a Tensor of shape (S)
- "label" is an integer defining the class annotation
- "video_name" is the video path stem
- "video_index" is the index for the video used by the video sampler
- "clip_index" is the index of the clip within the video (defined by clip sampler)


Transform functions can either be implemented by the user application or reused from any library that's domain specific to the modality. E.g. for video we recommend using TorchVision, for audio we recommend TorchAudio.

PyTorchVideo also provides several transforms which you can see in the docs (TODO link to transform docs). For example, the library provides several helper transforms that improve interoperability between other domain specific libraries. Notably:
  - ApplyTransformToKey(key, transform) - applies a transform to specific modality
  - RemoveKey(key) - remove a specific modality from the clip

Below we revise the LightningDataModule from the last section to include transforms coming from both TorchVision and PyTorchVideo. For brevity we'll just show the KineticsDataModule.train_dataloader method. The validation transform be the same just without the augmentations (RandomShortSideScale, RandomCropVideo, RandomHorizontalFlipVideo).

```
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import Compose
from torchvision.transforms.transforms_video import ( # TODO change these to non transform_video transforms
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)

  ...

    def train_dataloader(self):
      """
      Create the Kinetics train partition from the list of video labels in {self._DATA_PATH}/train.csv.
      Add transform that subsamples and normalizes the video before applying the scale, crop and flip
      augmentations.
      """
      train_transform = Compose(
          [
              ApplyTransformToKey(
                  key="video",
                  transform=Compose(
                      [
                          UniformTemporalSubsample(8),
                          NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                          RandomShortSideScale(min_size=256, max_size=320),
                          RandomCropVideo(244),
                          RandomHorizontalFlipVideo(p=0.5),
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

  ...

```

# Model

A "flat" interface is provided to allow complex PyTorchVideo models to be configured with ease. To build a 3D ResNet the code below is all that's needed for a default version. See the docs for other configuration options. (TODO link to model docs).

```

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

To finalize the tutorial, let's create a pytorch_lightning.LightningModule. This defines the train and validation step code (i.e. the code inside the training and evaluation loops), and the optimizer.

```
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

      # The model expects a video tensor of shape (B, C, T, H, W), which is the format given by
      # the dataset
      y_hat = self.model(batch["video"])

      # Compute cross entropy loss, backwards is called by PyTorchLightning after being returned
      # from this method.
      loss = F.cross_entropy(y_hat, batch["label"])

      #  log the train loss to tensorboard
      self.log("train_loss", loss.item())

      return loss

  def validation_step(self, batch, batch_idx):
      y_hat = self.model(batch["video"])
      loss = F.cross_entropy(y_hat, batch["label"])
      self.log("val_loss", loss)
      return loss

  def configure_optimizers(self):
      """
      We use the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)
```

This VideoClassificationLightningModule and the previously built KineticsDataModule can be trained together using the pytorch_lightning.Trainer. The trainer class has many arguments to define the training environment (e.g. num_gpus, distributed training). To keep things simple we'll just use the default local cpu training but note that this would likely take weeks to train so you might want to use more performant settings.

```
	classification_module = VideoClassificationLightningModule()
	data_module = KineticsDataModule()
	trainer = pytorch_lightning.Trainer()
	trainer.fit(classification_module, data_module)
```

# Conclusion

In this tutorial we showed how to train a 3D ResNet on Kinetics using PyTorch Lightning. You can see the final code from the tutorial (as well as a few extra bells and whistles) here. 

To learn more about PyTorchVideo, check out the rest of the documentation and tutorials
