import itertools
from pathlib import Path
from random import shuffle
from shutil import unpack_archive
from typing import Tuple

import pytorch_lightning as pl
import requests
import torch
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class LabeledVideoDataModule(pl.LightningDataModule):

    SOURCE_URL: str = None
    SOURCE_DIR_NAME: str = ""
    NUM_CLASSES: int = 700
    VERIFY_SSL: bool = True

    def __init__(
        self,
        root: str = "./",
        clip_duration: int = 2,
        video_num_subsampled: int = 8,
        video_crop_size: int = 224,
        video_means: Tuple[float] = (0.45, 0.45, 0.45),
        video_stds: Tuple[float] = (0.225, 0.225, 0.225),
        video_min_short_side_scale: int = 256,
        video_max_short_side_scale: int = 320,
        video_horizontal_flip_p: float = 0.5,
        batch_size: int = 4,
        workers: int = 4,
        **kwargs
    ):

        super().__init__()
        self.root = root
        self.data_path = Path(self.root) / self.SOURCE_DIR_NAME
        self.clip_duration = clip_duration
        self.video_num_subsampled = video_num_subsampled
        self.video_crop_size = video_crop_size
        self.video_means = video_means
        self.video_stds = video_stds
        self.video_min_short_side_scale = video_min_short_side_scale
        self.video_max_short_side_scale = video_max_short_side_scale
        self.video_horizontal_flip_p = video_horizontal_flip_p
        self.batch_size = batch_size
        self.workers = workers

        # Transforms applied to train dataset
        self.train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.video_num_subsampled),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.video_means, self.video_stds),
                    RandomShortSideScale(
                        min_size=self.video_min_short_side_scale,
                        max_size=self.video_max_short_side_scale,
                    ),
                    RandomCrop(self.video_crop_size),
                    RandomHorizontalFlip(p=self.video_horizontal_flip_p),
                ]
            ),
        )

        # Transforms applied on val dataset or for inference
        self.val_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.video_num_subsampled),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.video_means, self.video_stds),
                    ShortSideScale(self.video_min_short_side_scale),
                    CenterCrop(self.video_crop_size),
                ]
            ),
        )

    def prepare_data(self):
        """Download the dataset if it doesn't already exist. This runs only on rank 0"""
        if not (self.SOURCE_URL is None or self.SOURCE_DIR_NAME is None):
            if not self.data_path.exists():
                download_and_unzip(self.SOURCE_URL, self.root, verify=self.VERIFY_SSL)

    def train_dataloader(self):
        self.train_dataset = LimitDataset(
            labeled_video_dataset(
                data_path=str(Path(self.data_path) / "train"),
                clip_sampler=make_clip_sampler("random", self.clip_duration),
                transform=self.train_transform,
                decode_audio=False,
                video_sampler=DistributedSampler
                if (self.trainer is not None and self.trainer.use_ddp)
                else RandomSampler,
            )
        )
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def val_dataloader(self):
        self.val_dataset = LimitDataset(
            labeled_video_dataset(
                data_path=str(Path(self.data_path) / "val"),
                clip_sampler=make_clip_sampler("uniform", self.clip_duration),
                transform=self.val_transform,
                decode_audio=False,
                video_sampler=DistributedSampler
                if (self.trainer is not None and self.trainer.use_ddp)
                else RandomSampler,
            )
        )
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.workers
        )


class UCF11DataModule(LabeledVideoDataModule):

    SOURCE_URL: str = "https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip"
    SOURCE_DIR_NAME: str = "action_youtube_naudio"
    NUM_CLASSES: int = 11
    VERIFY_SSL: bool = False

    def __init__(self, **kwargs):
        """
        The UCF11 Dataset contains 11 action classes: basketball shooting, biking/cycling, diving,
        golf swinging, horse back riding, soccer juggling, swinging, tennis swinging, trampoline jumping,
        volleyball spiking, and walking with a dog.

        For each class, the videos are grouped into 25 group/scene folders containing at least 4 video clips each.
        The video clips in the same scene folder share some common features, such as the same actor, similar
        background, similar viewpoint, and so on.

        The folder structure looks like the following:

        /data_dir
        ├── basketball                     # Class Folder Path
        │   ├── v_shooting_01              # Scene/Group Folder Path
        │   │   ├── v_shooting_01_01.avi   # Video Path
        │   │   ├── v_shooting_01_02.avi
        │   │   ├── v_shooting_01_03.avi
        │   │   ├── ...
        │   ├── v_shooting_02
        │   ├── v_shooting_03
        │   ├── ...
        │   ...
        ├── biking
        │   ├── v_biking_01
        │   │   ├── v_biking_01_01.avi
        │   │   ├── v_biking_01_02.avi
        │   │   ├── v_biking_01_03.avi
        │   ├── v_biking_02
        │   ├── v_biking_03
        │   ...
        ...

        We take 80% of all scenes and use the videos within for training. The remaining scenes' videos
        are used for validation. We do this so the validation data contains only videos from scenes/actors
        that the model has not seen yet.
        """
        super().__init__(**kwargs)

    def setup(self, stage=None):
        """Set up anything needed for initializing train/val datasets. This runs on all nodes"""

        # Names of classes to predict
        # Ex. ['basketball', 'biking', 'diving', ...]
        self.classes = sorted(x.name for x in self.data_path.glob("*") if x.is_dir())

        # Mapping from label to class id.
        # Ex. {'basketball': 0, 'biking': 1, 'diving': 2, ...}
        self.label_to_id = {}

        # A list to hold all available scenes across all classes
        scene_folders = []

        for class_id, class_name in enumerate(self.classes):

            self.label_to_id[class_name] = class_id

            # The path of a class folder within self.data_path
            # Ex. 'action_youtube_naudio/{basketball|biking|diving|...}'
            class_folder = self.data_path / class_name

            # Collect scene folders within this class
            # Ex. 'action_youtube_naudio/basketball/v_shooting_01'
            for scene_folder in filter(Path.is_dir, class_folder.glob("v_*")):
                scene_folders.append(scene_folder)

        # Randomly shuffle the scene folders before splitting them into train/val
        shuffle(scene_folders)

        # Determine number of scenes in train/validation splits.
        self.num_train_scenes = int(0.8 * len(scene_folders))
        self.num_val_scenes = len(scene_folders) - self.num_train_scenes

        # Collect train/val paths to videos within each scene folder.
        # Validation only uses videos from scenes not seen by model during training
        self.train_paths = []
        self.val_paths = []
        for i, scene_path in enumerate(scene_folders):

            # The actual name of the class (Ex. 'basketball')
            class_name = scene_path.parent.name

            # Loop over all the videos within the given scene folder.
            for video_path in scene_path.glob("*.avi"):

                # Construct a tuple containing (<path to a video>, <dict containing extra attributes/metadata>)
                # In our case, we assign the class's ID as 'label'.
                labeled_path = (video_path, {"label": self.label_to_id[class_name]})

                if i < self.num_train_scenes:
                    self.train_paths.append(labeled_path)
                else:
                    self.val_paths.append(labeled_path)

    def train_dataloader(self):
        self.train_dataset = LimitDataset(
            LabeledVideoDataset(
                self.train_paths,
                clip_sampler=make_clip_sampler("random", self.clip_duration),
                decode_audio=False,
                transform=self.train_transform,
                video_sampler=RandomSampler,
            )
        )
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.workers
        )

    def val_dataloader(self):
        self.val_dataset = LimitDataset(
            LabeledVideoDataset(
                self.val_paths,
                clip_sampler=make_clip_sampler("uniform", self.clip_duration),
                decode_audio=False,
                transform=self.val_transform,
                video_sampler=RandomSampler,
            )
        )
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.workers
        )


def download_and_unzip(url, data_dir="./", verify=True):
    """Download a zip file from a given URL and unpack it within data_dir.

    Args:
        url (str): A URL to a zip file.
        data_dir (str, optional): Directory where the zip will be unpacked. Defaults to "./".
        verify (bool, optional): Whether to verify SSL certificate when requesting the zip file. Defaults to True.
    """
    data_dir = Path(data_dir)
    zipfile_name = url.split("/")[-1]
    data_zip_path = data_dir / zipfile_name
    data_dir.mkdir(exist_ok=True, parents=True)

    if not data_zip_path.exists():
        resp = requests.get(url, verify=verify)

        with data_zip_path.open("wb") as f:
            f.write(resp.content)

    unpack_archive(data_zip_path, extract_dir=data_dir)


class LimitDataset(torch.utils.data.Dataset):

    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
