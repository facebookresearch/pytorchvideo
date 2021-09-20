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

All the Kinetics models in the Model Zoo are trained and tested with the same data as [Non-local Network](https://github.com/facebookresearch/video-nonlocal-net/blob/main/DATASET.md) and [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md). For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).


### Charades

We follow [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) to prepare the Charades dataset as follow:

1. Download the Charades RGB frames from [official website](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/val.csv)).


### Something-Something V2

We follow [PySlowFast](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) to prepare the Something-Something V2 dataset as follow:

1. Download the dataset and annotations from [official website](https://20bn.com/datasets/something-something).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

3. Extract the frames from downloaded videos at 30 FPS. We used ffmpeg-4.1.3 with command:
    ```
    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    ```
4. The extracted frames should be organized to be consistent with the paths in frame lists.


### AVA (Actions V2.2)

The AVA Dataset could be downloaded from the [official site](https://research.google.com/ava/download.html#ava_actions_download)

We followed the same [downloading and preprocessing procedure](https://github.com/facebookresearch/video-long-term-feature-banks/blob/main/DATASET.md) as the [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/abs/1812.05038) do.

You could follow these steps to download and preprocess the data:

1. Download videos

```
DATA_DIR="../../data/ava/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
```

2. Cut each video from its 15th to 30th minute. AVA has valid annotations only in this range.

```
IN_DATA_DIR="../../data/ava/videos"
OUT_DATA_DIR="../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
```

3. Extract frames

```
IN_DATA_DIR="../../data/ava/videos_15min"
OUT_DATA_DIR="../../data/ava/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
```

4. Download annotations

```
DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_v2.2.zip -P ${DATA_DIR}
unzip -q ${DATA_DIR}/ava_v2.2.zip -d ${DATA_DIR}
```

5. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).

6. Download person boxes that are generated using a person detector trained on AVA - ([train](https://dl.fbaipublicfiles.com/pytorchvideo/data/ava/ava_detection_test.csv), [val](https://dl.fbaipublicfiles.com/pytorchvideo/data/ava/ava_detection_val.csv), [test](https://dl.fbaipublicfiles.com/pytorchvideo/data/ava/ava_detection_test.csv)) and put them in the `annotations` folder (see structure above). Copy files to the annotations directory mentioned in step 4. 
If you prefer to use your own person detector, please generate detection predictions files in the suggested format in step 6.

Download the ava dataset with the following structure:

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```
