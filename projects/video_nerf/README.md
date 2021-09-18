# Train a NeRF model with PyTorchVideo and PyTorch3D

This project demonstrates how to use the video decoder from PyTorchVideo to load frames from a video of an object from the [Objectron dataset](https://github.com/google-research-datasets/Objectron), and use this to train a NeRF [1] model with [PyTorch3D](https://github.com/facebookresearch/pytorch3d). Instead of decoding and storing all the video frames as images, PyTorchVideo offers an easy alternative to load and access frames on the fly.  For this project we will be using the [NeRF implementation from PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf).

### Set up

#### Installation

Install PyTorch3D

```python
# Create new conda environment
conda create -n 3ddemo
conda activate 3ddemo

# Install PyTorch3D
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.1
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install pytorch3d -c pytorch3d-nightly
```

Install PyTorchVideo if you haven't installed it already (assuming you have cloned the repo locally):

```python
cd pytorchvideo
python -m pip install -e .
```

Install some extras libraries needed for NeRF:

```python
pip install visdom Pillow matplotlib tqdm plotly
pip install hydra-core --upgrade
```

#### Set up NeRF Model

We will be using the PyTorch3D NeRF implementation. We have already installed the PyTorch3d conda packages, so now we only need to clone the NeRF implementation:

```python
cd pytorchvideo/tutorials/video_nerf
git clone https://github.com/facebookresearch/pytorch3d.git
cp -r pytorch3d/projects/nerf .

# Remove the rest of the PyTorch3D repo
rm -r pytorch3d
```

#### Dataset

###### Download the Objectron repo

The repo contains helper functions for reading the metadata files. Clone it to the path `pytorchvideo/tutorials/video_nerf/Objectron`.

```python
git clone https://github.com/google-research-datasets/Objectron.git

# Also install protobuf for parsing the metadata
pip install protobuf
```

###### Download an example video

For this demo we will be using a short video of a chair from the [Objectron dataset](https://github.com/google-research-datasets/Objectron). Each video is accompanied by metadata with the camera parameters for each frame. You can download an example video for a chair and the associated metadata by running the following script:

```python
python download_objectron_data.py
```

The data files will be downloaded to the path: `pytorchvideo/tutorials/video_nerf/nerf/data/objectron`. Within the script you can change the index of the video to use to obtain a different chair video.  We will create and save a random split of train/val/test when the video is first loaded by the NeRF model training script.

Most of the videos are recorded in landscape mode with image size (H, W) = [1440, 1920].


#### Set up new configs

For this dataset we need a new config file and data loader to use it with the PyTorch3D NeRF implementation. Copy the relevant dataset and config files into the `nerf` folder and replace the original files:

```python
# Make sure you are at the path: pytorchvideo/tutorials/video_nerf
# Rename the current dataset file
mv nerf/nerf/dataset.py nerf/nerf/nerf_dataset.py

# Move the new objectron specific files into the nerf folder
mv dataset.py nerf/nerf/dataset.py
mv dataset_utils.py nerf/nerf/dataset_utils.py
mv objectron.yaml nerf/configs
```

In the `video_dataset.py` file we use the PyTorchVideo `EncodedVideo` class to load a video `.MOV` file, decode it into frames and access the frames by the index.

#### Train model

Run the model training:

```python
cd nerf
python ./train_nerf.py --config-name objectron
```

#### Visualize predictions

Predictions and metrics will be logged to Visdom. Before training starts launch the visdom server:

```python
python -m visdom.server
```

Navigate to `https://localhost:8097` to view the logs and visualizations.

After training, you can generate predictions on the test set:

```python
python test_nerf.py --config-name objectron test.mode='export_video' data.image_size="[96,128]"
```

For a higher resolution video you can increase the image size to e.g. [192, 256] (note that this will slow down inference).

You will need to specify the `scene_center` for the video in the `objectron.yaml` file. This is set for the demo video specified in `download_objectron_data.py`. For a different video you can calculate the scene center inside [`eval_video_utils.py`](https://github.com/facebookresearch/pytorch3d/blob/main/projects/nerf/nerf/eval_video_utils.py#L99). After line 99 you can add the following code to compute the center:

```python
# traj is the circular camera trajectory on the camera mean plane.
# We want the camera to always point towards the center of this trajectory.
x_center = traj[..., 0].mean().item()
z_center = traj[..., 2].mean().item()
y_center = traj[0, ..., 1]
scene_center = [x_center, y_center, z_center]
```
You can also point the camera down/up relative to the camera mean plane e.g. `y_center -= 0.5`

Here is an example of a video reconstruction generated using a trained NeRF model. NOTE: the quality of reconstruction is highly dependent on the camera pose range and accuracy in the annotations - try training a model for a few different chairs in the dataset to see which one has the best results.

<img src="https://dl.fbaipublicfiles.com/pytorchvideo/projects/pytorch3d-nerf/chair.gif" width="310"/>

##### References
[1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng, NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV2020
