---
id: tutorial_torchhub_detection_inference
title: Running a pre-trained PyTorchVideo classification model using Torch Hub
---

# Introduction

PyTorchVideo provides several pretrained models through [Torch Hub](https://pytorch.org/hub/). In this tutorial we will show how to load a pre trained video classification model in PyTorchVideo and run it on a test video. The PyTorchVideo Torch Hub models were trained on the Kinetics 400 dataset and finetuned specifically for detection on AVA v2.2 dataset.  Available models are described in [model zoo documentation](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html).

NOTE: Currently, this tutorial only works if ran on local clone from the directory `pytorchvideo/tutorials/video_detection_example`

This tutorial assumes that you have installed [Detectron2]((https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)) and [Opencv-python](https://pypi.org/project/opencv-python/) on your machine.

# Imports
```python
from functools import partial
import numpy as np

import cv2
import torch

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection # Another option is slowfast_r50_detection

from visualization import VideoVisualizer
```

# Load Model using Torch Hub API
PyTorchVideo provides several pretrained models through Torch Hub. Available models are described in [model zoo documentation.](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md)

Here we are selecting the slow_r50_detection model which was trained using a 4x16 setting on the Kinetics 400 dataset and fine tuned on AVA V2.2 actions dataset.

NOTE: to run on GPU in Google Colab, in the menu bar selet: Runtime -> Change runtime type -> Harware Accelerator -> GPU

```python
device = 'cuda' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)
```

# Load an off-the-shelf Detectron2 object detector

We use the object detector to detect bounding boxes for the people.
These bounding boxes later feed into our video action detection model.
For more details, please refer to the Detectron2's object detection tutorials.

To install Detectron2, please follow the instructions mentioned [here](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes
```

# Define the transformations for the input required by the model
Before passing the video and bounding boxes into the model we need to apply some input transforms and sample a clip of the correct frame rate in the clip.

Here, below we define a method that can pre-process the clip and bounding boxes. It generates inputs accordingly for both Slow (Resnet) and SlowFast models depending on the parameterization of the variable `slow_fast_alpha`.

```python
def ava_inference_transform(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes
```

# Setup

Download the id to label mapping for the AVA V2.2 dataset on which the Torch Hub models were finetuned.
This will be used to get the category label names from the predicted class ids.

Create a visualizer to visualize and plot the results(labels + bounding boxes).

```python
# Dowload the action text to id mapping
!wget https://dl.fbaipublicfiles.com/pytorchvideo/data/class_names/ava_action_list.pbtxt

# Create an id to label name mapping
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
# Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)
```

# Load an example video
We get an opensourced video off the web from WikiMedia.
```python
# Download the demo video.
!wget https://dl.fbaipublicfiles.com/pytorchvideo/projects/theatre.webm

# Load the video
encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path('theatre.webm')
print('Completed loading encoded video.')
```

# Get model predictions

Generate bounding boxes and action predictions for a 10 second clip in the video.

```python
# Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
time_stamp_range = range(90,100) # time stamps in video for which clip is sampled.
clip_duration = 1.0 # Duration of clip used for each inference step.
gif_imgs = []

for time_stamp in time_stamp_range:
    print("Generating predictions for time stamp: {} sec".format(time_stamp))

    # Generate clip around the designated time stamps
    inp_imgs = encoded_vid.get_clip(
        time_stamp - clip_duration/2.0, # start second
        time_stamp + clip_duration/2.0  # end second
    )
    inp_imgs = inp_imgs['video']

    # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
    # We use the the middle image in each clip to generate the bounding boxes.
    inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
    inp_img = inp_img.permute(1,2,0)

    # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print("Skipping clip no frames detected at time stamp: ", time_stamp)
        continue

    # Preprocess clip and bounding boxes for video action recognition.
    inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
    # Prepend data sample id for each bounding box.
    # For more details refere to the RoIAlign in Detectron2
    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)

    # Generate actions predictions for the bounding boxes in the clip.
    # The model here takes in the pre-processed video clip and the detected bounding boxes.
    preds = video_model(inputs.unsqueeze(0).to(device), inp_boxes.to(device))


    preds= preds.to('cpu')
    # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
    preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

    # Plot predictions on the video and save for later visualization.
    inp_imgs = inp_imgs.permute(1,2,3,0)
    inp_imgs = inp_imgs/255.0
    out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
    gif_imgs += out_img_pred

print("Finished generating predictions.")
```

We now save the predicted video containing bounding boxes and action labels for the bounding boxes.

```python
height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

vide_save_path = 'output.mp4'
video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))

for image in gif_imgs:
    img = (255*image).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video.write(img)
video.release()

print('Predictions are saved to the video file: ', vide_save_path)
```

# Conclusion

In this tutorial we showed how to load and run a pretrained PyTorchVideo detection model on a test video. You can run this tutorial as a notebook in the PyTorchVideo tutorials directory.

To learn more about PyTorchVideo, check out the rest of the [documentation](https://pytorchvideo.readthedocs.io/en/latest/index.html)  and [tutorials](https://pytorchvideo.org/docs/tutorial_overview).
