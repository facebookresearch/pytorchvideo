import requests
from argparse import Namespace, ArgumentParser
import pytorch_lightning
from pathlib import Path
from shutil import unpack_archive
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.data import LabeledVideoDataset

from torch.utils.data import DistributedSampler, RandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset
import torch
import itertools
from torch.utils.data import DataLoader
from random import shuffle


class LabeledVideoDataModule(pytorch_lightning.LightningDataModule):

    TRAIN_PATH = "train.csv"
    VAL_PATH = "val.csv"
    SOURCE_URL = None
    SOURCE_DIR_NAME = None

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.root = Path(self.args.data_path) / self.SOURCE_DIR_NAME
        if not (self.SOURCE_URL is None or self.SOURCE_DIR_NAME is None):
            if not self.root.exists():
                download_and_unzip(self.SOURCE_URL, self.args.data_path, verify=getattr(self.args, 'verify', True))

    def _make_transforms(self, mode: str):

        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        elif self.args.data_type == "audio":
            transform = [
                self._audio_transform(),
                RemoveKey("video"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.video_num_subsampled),
                    Normalize(args.video_means, args.video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=args.video_min_short_side_scale,
                            max_size=args.video_max_short_side_scale,
                        ),
                        RandomCrop(args.video_crop_size),
                        RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
            ),
        )

    def _audio_transform(self):
        args = self.args
        n_fft = int(float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size)
        hop_length = int(float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size)
        eps = 1e-10
        return ApplyTransformToKey(
            key="audio",
            transform=Compose(
                [
                    Resample(
                        orig_freq=args.audio_raw_sample_rate,
                        new_freq=args.audio_resampled_rate,
                    ),
                    MelSpectrogram(
                        sample_rate=args.audio_resampled_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=args.audio_num_mels,
                        center=False,
                    ),
                    Lambda(lambda x: x.clamp(min=eps)),
                    Lambda(torch.log),
                    UniformTemporalSubsample(args.audio_mel_num_subsample),
                    Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                    Lambda(lambda x: x.view(1, x.size(0), 1, x.size(1))),  # (T, F) -> (1, T, 1, F)
                    Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
                ]
            ),
        )

    def _make_ds_and_loader(self, mode: str):
        ds = LimitDataset(
            labeled_video_dataset(
                data_path=str(Path(self.root) / (self.TRAIN_PATH if mode == 'train' else self.VAL_PATH)),
                clip_sampler=make_clip_sampler("random" if mode == 'train' else 'uniform', self.args.clip_duration),
                video_path_prefix=self.args.video_path_prefix,
                transform=self._make_transforms(mode=mode),
                video_sampler=DistributedSampler if (self.trainer is not None and self.trainer.use_ddp) else RandomSampler,
            )
        )
        return ds, DataLoader(ds, batch_size=self.args.batch_size, num_workers=self.args.workers)

    def train_dataloader(self):
        self.train_dataset, loader = self._make_ds_and_loader('train')
        return loader

    def val_dataloader(self):
        self.val_dataset, loader = self._make_ds_and_loader('val')
        return loader


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
        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


class KineticsDataModule(LabeledVideoDataModule):
    TRAIN_PATH = 'train.csv'
    VAL_PATH = 'val.csv'
    NUM_CLASSES = 700


class MiniKineticsDataModule(LabeledVideoDataModule):

    TRAIN_PATH = "train"
    VAL_PATH = "val"
    SOURCE_URL = "https://pl-flash-data.s3.amazonaws.com/kinetics.zip"
    SOURCE_DIR_NAME = 'kinetics'
    NUM_CLASSES = 6


class UCF11DataModule(LabeledVideoDataModule):
    TRAIN_PATH = None
    VAL_PATH = None
    SOURCE_URL = "https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip"
    SOURCE_DIR_NAME = 'action_youtube_naudio'
    NUM_CLASSES = 11

    def __init__(self, args):
        args.verify = False
        super().__init__(args)

        data_path = Path(self.args.data_path)
        root = data_path / self.SOURCE_DIR_NAME
        self.classes = [x.name for x in root.glob("*") if x.is_dir()]
        self.id_to_label = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_label = {v: k for k, v in self.id_to_label.items()}
        self.num_classes = len(self.classes)

        self.train_paths = []
        self.val_paths = []
        self.holdout_scenes = {}
        for c in self.classes:

            # Scenes within each class directory
            scene_names = sorted(x.name for x in (root / c).glob("*") if x.is_dir() and x.name != 'Annotation')
            shuffle(scene_names)

            # Holdout a random actor/scene
            holdout_scene = scene_names[-1]
            scene_names = scene_names[:-1]

            # Keep track of which scenes we held out for each class w/ a dict
            self.holdout_scenes[c] = holdout_scene

            for v in (root / c).glob('**/*.avi'):
                labeled_path = (v, {"label": self.class_to_label[c]})
                if v.parent.name != holdout_scene:
                    self.train_paths.append(labeled_path)
                else:
                    self.val_paths.append(labeled_path)


    def _make_ds_and_loader(self, mode: str):
        ds = LimitDataset(
            LabeledVideoDataset(
                self.train_paths if mode == 'train' else self.val_paths,
                clip_sampler=make_clip_sampler("random" if mode == 'train' else 'uniform', self.args.clip_duration),
                decode_audio=False,
                transform=self._make_transforms(mode=mode),
                video_sampler=DistributedSampler if (self.trainer is not None and self.trainer.use_ddp) else RandomSampler,
            )
        )
        return ds, DataLoader(ds, batch_size=self.args.batch_size, num_workers=self.args.workers)


def download_and_unzip(url, data_dir="./", verify=True):
    data_dir = Path(data_dir)
    zipfile_name = url.split("/")[-1]
    data_zip_path = data_dir / zipfile_name
    data_dir.mkdir(exist_ok=True, parents=True)

    if not data_zip_path.exists():
        resp = requests.get(url, verify=verify)

        with data_zip_path.open("wb") as f:
            f.write(resp.content)

    unpack_archive(data_zip_path, extract_dir=data_dir)


if __name__ == "__main__":
    args = parse_args('--batch_size 4 --data_path ./yt_data'.split())
    dm = UCF11DataModule(args)
