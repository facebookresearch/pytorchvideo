## Data Preparation

### Kinetics

For more information about Kinetics dataset, please refer the official [website](https://deepmind.com/research/open-source/kinetics). You can take the following steps to prepare the dataset:

1. Download the videos via the official [scripts](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

2. Preprocess the downloaded videos by resizing to the short edge size of 256.

3. Prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

All the Kinetics models in the Model Zoo are trained and tested with the same data as [Non-local Network](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md) and [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md). For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).


### Charades

We follow [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) to prepare the Charades dataset as follow:

1. Download the Charades RGB frames from [official website](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/val.csv)).


### Something-Something V2

We follow [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) to prepare the Something-Something V2 dataset as follow:

1. Download the dataset and annotations from [official website](https://20bn.com/datasets/something-something).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

3. Extract the frames from downloaded videos at 30 FPS. We used ffmpeg-4.1.3 with command:
    ```
    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    ```
4. The extracted frames should be organized to be consistent with the paths in frame lists.
