---
id: tutorial_torchhub_inference
title: Running a pre-trained PyTorchVideo classification model using Torch Hub
---

# Introduction

PyTorchVideo provides several pretrained models through [Torch Hub](https://pytorch.org/hub/). In this tutorial we will show how to load a pre trained video classification model in PyTorchVideo and run it on a test video. The PyTorchVideo Torch Hub models were trained on the Kinetics 400 [1] dataset.  Available models are described in [model zoo documentation](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html).

[1] W. Kay, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

NOTE: Currently, this tutorial will only work with a local clone of the PyTorchVideo GitHub repo.

# Imports

```python
import json
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
```

# Load Model

Let's select the `slow_r50` model which was trained using a 8x8 setting on the Kinetics 400 dataset.

```python
# Device on which to run the model
device = "cuda:0"

# Pick a pretrained model
model_name = "slow_r50"

# Local path to the parent folder of hubconf.py in the pytorchvideo codebase
path = '../'
model = torch.hub.load(path, source="local", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.eval()
model = model.to(device)
```

# Setup Labels

Next let's download the id-to-label mapping for the Kinetics 400 dataset on which the torch hub models were trained. This will be used to get the category label names from the predicted class ids.

```python
!wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json
```

```python
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")
```

# Input Transform

Before passing the video into the model we need to apply some input transforms and sample a clip of the correct duration. We will define them below.

```python
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
# If you want to try another of the torch hub models you will need to modify this transform
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second
```

# Load an example video
We can now test the model with an example video from the Kinetics validation set such as this [archery video](https://www.youtube.com/watch?v=3and4vWkW4s).

We will load the video and apply the input transform.


```python
# Download the example video file
!wget https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4
```

```python
# Load the example video
video_path = "archery.mp4"

# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = inputs.to(device)
```

### Get model predictions

Now we are ready to pass the input into the model and classify the action.

```python
# Pass the input clip through the model
preds = model(inputs[None, ...])
```

Let's look at the top 5 best predictions:

```python
# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
print("Predicted labels: %s" % ", ".join(pred_class_names))
```

# Conclusion

In this tutorial we showed how to load and run a pretrained PyTorchVideo model on a test video. You can run this tutorial as a notebook in the PyTorchVideo tutorials directory.

To learn more about PyTorchVideo, check out the rest of the [documentation](https://pytorchvideo.readthedocs.io/en/latest/index.html)  and [tutorials](https://pytorchvideo.org/docs/tutorial_overview).
