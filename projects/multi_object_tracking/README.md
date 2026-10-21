# Multi Object Tracking for PyTorchVideo

The project demonstrates the use of multi object tracking for PuTorchVideo.
Currently the project contains JDETracker as the multiobject tracker. Very soon,more trackers can
be added to this project.


## JDETracker
**Joint Detection and Embedding (JDE) Tracker** was introduced in the paper 'Towards Real-Time Multi-Object Tracking' (https://arxiv.org/abs/1909.12605).
This tracker can work with any model which can provide the following two inputs:

(i) pred_dets: Detection results of the image i.e. x1, y1, x2, y2, object_conf (The detections should/could be passed through NMS or a similar technique followed by scaling the coordinates to the original image size -  _This way allowing more flexibility for the tracker_)

(ii) pred_embs: Embedding results of the image.

#### Folder structure
    .
    ├── mot                     # The main multi-object tracking library which can be later on added to PyTorchVideo
    │   ├── matching            
    │   |   ├── ..
    |   |   ├── jde_matching.py 
    │   ├── motion
    |   ├── ..
    |   |   ├── kalman_filter.py # File related to Kalman Filter 
    │   ├── tracker 
    |   ├── ..
    |   |   ├── base_jde.py     # Contains TrackState and Strack classes which are the core base classes for JDE Tracking 
    |   |   ├── jde_tracker.py  # Contains the class to be used using JDE Tracking  
    ├── demo_jde_tracker.py     # Sample demo file for using a detector and integrating it with JDE tracker to obtain tracking results.
    ├── detector_utils.py       # utility file (including the model definition) related to the detector used in the paper
    ├── tests                   # test files related to JDETracker
    ├── jde_dets.pt             # file containing detections, embedding for running test file
    └── README.md

#### Set up

In order to set up your system for testing the JDETracker with integrated with a detector - these are the required steps:

(1) Set the current working directory to the multi_object_tracking folder:
```python
cd pytorchvideo/projects/multi_object_tracking
```

(1) Download weights:
Download the weights from this location https://drive.google.com/open?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA and copy it in the 'weights' folder 
```python
mkdir weights
cd weights
cp path/to/weights . #copy weights file to this folder
cd ..
```

(2) Download any sample video:
For this current demo we will take the MOT16-03.mp4 video which is available on the link: https://drive.google.com/file/d/1254q3ruzBzgn4LUejDVsCtT05SIEieQg/view?usp=sharing
This video is stored at videos/MOT16-03.mp4

(3) Set up the requirements:
* [Pytorch](https://pytorch.org) >= 1.2.0 
* python-opencv
* cython-bbox (`pip install cython_bbox`)
* ffmpeg (used for creating a tracking result video)

#### Running the demo 

```python
python demo_jde_tracker.py 
--input-video videos/MOT16-03.mp4
--weights weights/jde.1088x608.uncertainty.pt
--cfg yolov3_1088x608.cfg   
--output-root .
```

* The results will be stored as follows:

(i) Individual frames with their bounding boxes and track ids will be stored in the _frame_ folder.

(ii) After the entire video is processed, a _result.mp4_ file will be created for the video of the entire run.

#### Running the JDE Tracker with any other detector
As stated above, The JDE tracker can be integrated with any model which can output both detections and embeddings.
Also refer to the test file in tests folder for futher reference
```python

# Step 1: Create instance of JDETracker
from mot.tracker import JDETracker
tracker = JDETracker

# Step 2:  (In a loop)
# Run image/video frame through the detector, obtain detections and embeddings
# Filter them through NMS and scale them back to the original image size.
# Lastly pass them to the update function of the JDETracker
# ..... 
online_targets = mot.tracker import JDETracker
# ....
```

##### References
[1] Zhongdao Wang, Liang Zheng, Yixuan Liu, Yali Li, Shengjin Wang, Towards Real-Time Multi-Object Tracking, ECCV 2020

[2] https://github.com/Zhongdao/Towards-Realtime-MOT
