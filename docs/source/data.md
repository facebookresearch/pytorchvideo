# Overview

PyTorchVideo datasets are subclasses of either [```torch.utils.data.Dataset```](https://pytorch.org/docs/stable/data.html#map-style-datasets) or [```torch.utils.data.IterableDataset```](https://pytorch.org/docs/stable/data.html#iterable-style-datasets). As such, they can all be used with a [```torch.utils.data.DataLoader```](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoade), which can load multiple samples in parallel using [```torch.multiprocessing```](https://pytorch.org/docs/stable/multiprocessing.html) workers. For example:

```python
dataset = pytorchvideo.data.Kinetics(
        data_path="path/to/kinetics_root/train.csv",
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", duration=2),
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8)
```

## How do PyTorchVideo datasets work?

Although there isn't a strict interface governing how PyTorchVideo datasets work, they all share a common design as follows:

1. Each dataset starts by taking a list of video paths and labels in some form. For example, Kinetics can take a file with each row containing a video path and label, or a directory containing a ```\<label\>/\<video_name\>.mp4``` like file structure. Each respective dataset documents the exact structure it expected for the given data path.

2. At each iteration a video sampler is used to determine which video-label pair is going to be sampled from the list of videos from the previous point. For some datasets this is required to be a random sampler, others reuse the [```torch.utils.data.Sampler```](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) interface for more flexibility.

3. A clip sampler is then used to determine which frames to sample from the selected video. For example, your application may want to sample 2 second clips at random for the selected video at each iteration. Some datasets like Kinetics make the most of the [```pytorchvideo.data.clip_sampling```](https://pytorchvideo.readthedocs.io/en/latest/api/data/extra.html#pytorchvideo-data-clip-sampling) interface to provide flexibility on how to define these clips. Other datasets simply require you to specify an enum for common clip sampling configurations.

4. Depending on if the underlying videos are stored as either encoded videos (e.g. mp4) or frame videos (i.e. a folder of images containing each decoded frame) - the video clip is then selectively read or decoded into the canonical video tensor with shape ```(C, T, H, W)``` and audio tensor with shape ```(S)```. We provide two options for decoding: PyAv or TorchVision, which can be chosen in the interface of the datasets that supported encoded videos.

5. The next step of a PyTorchVideo dataset is creating a clip dictionary containing the video modalities, label and metadata ready to be returned. An example clip dictionary might look like this:
    ```
      {
         'video': <video_tensor>,     # Shape: (C, T, H, W)
         'audio': <audio_tensor>,     # Shape: (S)
         'label': <action_label>,     # Integer defining class annotation
         'video_name': <video_path>,  # Video file path stem
         'video_index': <video_id>,   # index of video used by sampler
         'clip_index': <clip_id>      # index of clip sampled within video
      }
    ```
    All datasets share the same canonical modality tensor shapes and dtypes, which aligns with tensor types of other domain specific libraries (e.g. TorchVision, TorchAudio).

6. The final step before returning a clip, involves feeding it into a transform callable that can be defined for of all PyTorchVideo datasets. This callable is used to allow custom data processing or augmentations to be applied before batch collation in the [```torch.utils.data.DataLoader```](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). PyTorchVideo provides common [```pytorchvideo.transforms```](https://pytorchvideo.readthedocs.io/en/latest/transforms.html) that are useful for this callable, but users can easily define their own too.

## Available datasets:

* Charades
* Domsev
* EpicKitchen
* HMDB51
* Kinetics
* SSV2
* UCF101
