# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import json
import logging
import os
from bisect import bisect_left
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np

import torch
import torch.autograd.profiler as profiler
import torch.utils.data
import torchaudio
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.ego4d.utils import (
    Ego4dImuDataBase,
    get_label_id_map,
    MomentsClipSampler,
)
from pytorchvideo.data.utils import get_logger
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Div255,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
)
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip

log: logging.Logger = get_logger("Ego4dMomentsDataset")


class Ego4dImuData(Ego4dImuDataBase):
    """
    Wrapper for Ego4D IMU data loads, assuming one csv per video_uid at the provided path.
    """

    def __init__(self, imu_path: str) -> None:
        """
        Args:
            imu_path (str):
                Base path to construct IMU csv file paths.
                i.e. <base_path>/<video_uid>.csv
        """
        assert imu_path

        self.path_imu = imu_path
        self.IMU_by_video_uid: Dict[str, Any] = {}
        for f in g_pathmgr.ls(self.path_imu):
            self.IMU_by_video_uid[f.split(".")[0]] = f.replace(".csv", "")

        log.info(
            f"Number of videos with IMU (before filtering) {len(self.IMU_by_video_uid)}"
        )

        self.imu_video_uid: Optional[str] = None
        self.imu_video_data: Optional[Tuple[np.ndarray, np.ndarray, int]] = None

    def has_imu(self, video_uid: str) -> bool:
        return video_uid in self.IMU_by_video_uid

    def _load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        with g_pathmgr.open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                data.append(row)
        return data

    def _load_imu(self, video_uid: str) -> Tuple[np.ndarray, np.ndarray, int]:
        file_path = os.path.join(self.path_imu, video_uid) + ".csv"
        data_csv = self._load_csv(file_path)
        data_IMU = defaultdict(list)
        for row in data_csv:
            for k, v in row.items():
                if v != "":
                    data_IMU[k].append(float(v))
                else:
                    data_IMU[k].append(0.0)
        signal = np.array(
            [
                data_IMU["accl_x"],
                data_IMU["accl_y"],
                data_IMU["accl_z"],
                data_IMU["gyro_x"],
                data_IMU["gyro_y"],
                data_IMU["gyro_z"],
            ]
        ).transpose()
        # normalize
        signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)
        timestamps = np.array(data_IMU["canonical_timestamp_ms"])
        sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
        if sampling_rate < 0:
            # regenerate timestamps with 198 hz
            new_timestamps = timestamps[0] + (1000 / 198) * np.arange(len(timestamps))
            timestamps = np.array(new_timestamps)
            sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
        return signal, timestamps, sampling_rate

    def _get_imu_window(
        self,
        window_start: float,
        window_end: float,
        signal: np.ndarray,
        timestamps: np.ndarray,
        sampling_rate: float,
    ) -> Dict[str, Any]:
        start_id = bisect_left(timestamps, window_start * 1000)
        end_id = bisect_left(timestamps, window_end * 1000)
        if end_id == len(timestamps):
            end_id -= 1

        sample_dict = {
            "timestamp": timestamps[start_id:end_id],
            "signal": signal[start_id:end_id],
            "sampling_rate": sampling_rate,
        }
        return sample_dict

    def get_imu(self, video_uid: str) -> Tuple[np.ndarray, np.ndarray, int]:
        # Caching/etc?
        return self._load_imu(video_uid)

    def get_imu_sample(
        self, video_uid: str, video_start: float, video_end: float
    ) -> Dict[str, Any]:
        # Assumes video clips are loaded sequentially, will lazy load imu
        if not self.imu_video_uid or video_uid != self.imu_video_uid:
            self.imu_video_uid = video_uid
            self.imu_video_data = self._load_imu(video_uid)
            assert self.imu_video_data
        imu_signal, timestamps, sampling_rate = self.imu_video_data

        return self._get_imu_window(
            video_start,
            video_end,
            imu_signal,
            timestamps,
            sampling_rate,
        )


class Ego4dMomentsDataset(LabeledVideoDataset):
    """
    Ego4d video/audio/imu dataset for the moments benchmark:
    `<https://ego4d-data.org/docs/benchmarks/episodic-memory/>`

    This dataset handles the parsing of frames, loading and clip sampling for the
    videos.

    IO utilizing :code:`iopath.common.file_io.PathManager` to support
    non-local storage uri's.
    """

    VIDEO_FPS = 30
    AUDIO_FPS = 48000

    def __init__(
        self,
        annotation_path: str,
        metadata_path: str,
        split: Optional[str] = None,
        decode_audio: bool = True,
        imu: bool = False,
        clip_sampler: Optional[ClipSampler] = None,
        video_sampler: Type[
            torch.utils.data.Sampler
        ] = torch.utils.data.SequentialSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        decoder: str = "pyav",
        filtered_labels: Optional[List[str]] = None,
        window_sec: int = 10,
        audio_transform_type: str = "melspectrogram",
        imu_path: str = None,
        label_id_map: Optional[Dict[str, int]] = None,
        label_id_map_path: Optional[str] = None,
        video_path_override: Optional[Callable[[str], str]] = None,
        video_path_handler: Optional[VideoPathHandler] = None,
        eligible_video_uids: Optional[Set[str]] = None,
    ) -> None:
        """
        Args:
            annotation_path (str):
                Path or URI to Ego4d moments annotations json (ego4d.json). Download via:
                `<https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md>`

            metadata_path (str):
                Path or URI to primary Ego4d metadata json (moments.json). Download via:
                `<https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md>`

            split (Optional[str]): train/val/test

            decode_audio (bool): If True, decode audio from video.

            imu (bool): If True, load IMU data.

            clip_sampler (ClipSampler):
                A standard PTV ClipSampler. By default, if not specified, `MomentsClipSampler`

            video_sampler (VideoSampler):
                A standard PTV VideoSampler.

            transform (Optional[Callable[[Dict[str, Any]], Any]]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.

                    The clip input is a dictionary with the following format:
                        {{
                            'video': <video_tensor>,
                            'audio': <audio_tensor>,
                            'imu': <imu_tensor>,
                            'start_time': <float>,
                            'stop_time': <float>
                        }}

                If transform is None, the raw clip output in the above format is
                returned unmodified.

            decoder (str): Defines what type of decoder used to decode a video within
                `LabeledVideoDataset`.

            filtered_labels (List[str]):
                Optional list of moments labels to filter samples for training.

            window_sec (int): minimum window size in s

            audio_transform_type: melspectrogram / spectrogram / mfcc

            imu_path (Optional[str]):
                Path to the ego4d IMU csv file.  Required if imu=True.

            label_id_map / label_id_map_path:
                A map of moments labels to consistent integer ids.  If specified as a path
                we expect a vanilla .json dict[str, int].  Exactly one must be specified.

            video_path_override ((str) -> str):
                An override for video paths, given the video_uid, to support downsampled/etc
                videos.

            video_path_handler (VideoPathHandler):
                Primarily provided as an override for `CachedVideoPathHandler`

        Example Usage:
            Ego4dMomentsDataset(
                annotation_path="~/ego4d_data/v1/annotations/moments.json",
                metadata_path="~/ego4d_data/v1/ego4d.json",
                split="train",
                decode_audio=True,
                imu=False,
            )
        """

        assert annotation_path
        assert metadata_path
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split '{split}' not supported for ego4d"
        self.split: str = split
        self.training: bool = split == "train"
        self.window_sec = window_sec
        self._transform_source = transform
        self.decode_audio = decode_audio
        self.audio_transform_type = audio_transform_type
        assert (label_id_map is not None) ^ (
            label_id_map_path is not None
        ), f"Either label_id_map or label_id_map_path required ({label_id_map_path} / {label_id_map})"  # noqa

        self.video_means = (0.45, 0.45, 0.45)
        self.video_stds = (0.225, 0.225, 0.225)
        self.video_crop_size = 224
        self.video_min_short_side_scale = 256
        self.video_max_short_side_scale = 320

        try:
            with g_pathmgr.open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            raise FileNotFoundError(
                f"{metadata_path} must be a valid metadata json for Ego4D"
            )

        self.video_metadata_map: Dict[str, Any] = {
            x["video_uid"]: x for x in metadata["videos"]
        }

        if not g_pathmgr.isfile(annotation_path):
            raise FileNotFoundError(f"{annotation_path} not found.")

        try:
            with g_pathmgr.open(annotation_path, "r") as f:
                moments_annotations = json.load(f)
        except Exception:
            raise FileNotFoundError(f"{annotation_path} must be json for Ego4D dataset")

        self.label_name_id_map: Dict[str, int]
        if label_id_map:
            self.label_name_id_map = label_id_map
        else:
            self.label_name_id_map = get_label_id_map(label_id_map_path)
            assert self.label_name_id_map

        self.num_classes: int = len(self.label_name_id_map)
        log.info(f"Label Classes: {self.num_classes}")

        self.imu_data: Optional[Ego4dImuDataBase] = None
        if imu:
            assert imu_path, "imu_path not provided"
            self.imu_data = Ego4dImuData(imu_path)

        video_uids = set()
        clip_uids = set()
        clip_video_map = {}
        labels = set()
        labels_bypassed = set()
        cnt_samples_bypassed = 0
        cnt_samples_bypassed_labels = 0
        samples = []

        for vid in moments_annotations["videos"]:
            video_uid = vid["video_uid"]
            video_uids.add(video_uid)
            vsplit = vid["split"]
            if split and vsplit != split:
                continue
            # If IMU, filter videos without IMU
            if self.imu_data and not self.imu_data.has_imu(video_uid):
                continue
            if eligible_video_uids and video_uid not in eligible_video_uids:
                continue
            for clip in vid["clips"]:
                clip_uid = clip["clip_uid"]
                clip_uids.add(clip_uid)
                clip_video_map[clip_uid] = video_uid
                clip_start_sec = clip["video_start_sec"]
                clip_end_sec = clip["video_end_sec"]
                for vann in clip["annotations"]:
                    for lann in vann["labels"]:
                        label = lann["label"]
                        labels.add(label)
                        start = lann["start_time"]
                        end = lann["end_time"]
                        # remove sample with same timestamp
                        if start == end:
                            continue
                        start_video = lann["video_start_time"]
                        end_video = lann["video_end_time"]
                        assert end_video >= start_video

                        if abs(start_video - (clip_start_sec + start)) > 0.5:
                            log.warning(
                                f"Suspect clip/video start mismatch: clip: {clip_start_sec:.2f} + {start:.2f} video: {start_video:.2f}"  # noqa
                            )

                        # filter annotation base on the existing label map
                        if filtered_labels and label not in filtered_labels:
                            cnt_samples_bypassed += 1
                            labels_bypassed.add(label)
                            continue
                        metadata = self.video_metadata_map[video_uid]

                        if metadata["is_stereo"]:
                            cnt_samples_bypassed += 1
                            continue

                        if video_path_override:
                            video_path = video_path_override(video_uid)
                        else:
                            video_path = metadata["manifold_path"]
                        if not video_path:
                            cnt_samples_bypassed += 1
                            log.error("Bypassing invalid video_path: {video_uid}")
                            continue

                        sample = {
                            "clip_uid": clip_uid,
                            "video_uid": video_uid,
                            "duration": metadata["duration_sec"],
                            "clip_video_start_sec": clip_start_sec,
                            "clip_video_end_sec": clip_end_sec,
                            "labels": [label],
                            "label_video_start_sec": start_video,
                            "label_video_end_sec": end_video,
                            "video_path": video_path,
                        }
                        assert (
                            sample["label_video_end_sec"]
                            > sample["label_video_start_sec"]
                        )

                        if self.label_name_id_map:
                            if label in self.label_name_id_map:
                                sample["labels_id"] = self.label_name_id_map[label]
                            else:
                                cnt_samples_bypassed_labels += 1
                                continue
                        else:
                            log.error("Missing label_name_id_map")
                        samples.append(sample)

        self.cnt_samples: int = len(samples)

        log.info(
            f"Loaded {self.cnt_samples} samples. Bypass: {cnt_samples_bypassed} Label Lookup Bypass: {cnt_samples_bypassed_labels}"  # noqa
        )

        for sample in samples:
            assert "labels_id" in sample, f"init: Sample missing labels_id: {sample}"

        if not clip_sampler:
            clip_sampler = MomentsClipSampler(self.window_sec)

        super().__init__(
            [(x["video_path"], x) for x in samples],
            clip_sampler,
            video_sampler,
            transform=self._transform_mm,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        if video_path_handler:
            self.video_path_handler = video_path_handler

    def check_IMU(self, input_dict: Dict[str, Any]) -> bool:
        if (
            len(input_dict["imu"]["signal"].shape) != 2
            or input_dict["imu"]["signal"].shape[0] == 0
            or input_dict["imu"]["signal"].shape[0] < 200
            or input_dict["imu"]["signal"].shape[1] != 6
        ):
            log.warning(f"Problematic Sample: {input_dict}")
            return True
        else:
            return False

    def _transform_mm(self, sample_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        log.info("_transform_mm")
        with profiler.record_function("_transform_mm"):
            video_uid = sample_dict["video_uid"]
            assert video_uid

            assert sample_dict["video"] is not None
            assert (
                "labels_id" in sample_dict
            ), f"Sample missing labels_id: {sample_dict}"

            video = sample_dict["video"]

            expected = int(self.VIDEO_FPS * self.window_sec)
            actual = video.size(1)
            if expected != actual:
                log.error(
                    f"video size mismatch: actual: {actual} expected: {expected} video: {video.size()} uid: {video_uid}",  # noqa
                    stack_info=True,
                )
                return None

            start = sample_dict["clip_start"]
            end = sample_dict["clip_end"]
            assert start >= 0 and end >= start

            if abs((end - start) - self.window_sec) > 0.01:
                log.warning(f"Invalid IMU time window: ({start}, {end})")

            if self.imu_data:
                sample_dict["imu"] = self.imu_data.get_imu_sample(
                    video_uid,
                    start,
                    end,
                )
                if self.check_IMU(sample_dict):
                    log.warning(f"Bad IMU sample: ignoring: {video_uid}")
                    return None

            sample_dict = self._video_transform()(sample_dict)

            if self.decode_audio:
                audio_fps = self.AUDIO_FPS
                sample_dict["audio"] = self._preproc_audio(
                    sample_dict["audio"], audio_fps
                )
                sample_dict["spectrogram"] = sample_dict["audio"]["spectrogram"]

            labels = sample_dict["labels"]
            one_hot = self.convert_one_hot(labels)
            sample_dict["labels_onehot"] = one_hot

            if self._transform_source:
                sample_dict = self._transform_source(sample_dict)

            log.info(
                f"Sample ({sample_dict['video_name']}): "
                f"({sample_dict['clip_start']:.2f}, {sample_dict['clip_end']:.2f}) "
                f" {sample_dict['labels_id']} | {sample_dict['labels']}"
            )

            return sample_dict

    # pyre-ignore
    def _video_transform(self):
        """
        This function contains example transforms using both PyTorchVideo and
        TorchVision in the same callable. For 'train' model, we use augmentations (prepended
        with 'Random'), for 'val' we use the respective deterministic function
        """

        assert (
            self.video_means
            and self.video_stds
            and self.video_min_short_side_scale > 0
            and self.video_crop_size > 0
        )

        video_transforms = ApplyTransformToKey(
            key="video",
            transform=Compose(
                # pyre-fixme
                [Div255(), Normalize(self.video_means, self.video_stds)]
                + [  # pyre-fixme
                    RandomShortSideScale(
                        min_size=self.video_min_short_side_scale,
                        max_size=self.video_max_short_side_scale,
                    ),
                    RandomCrop(self.video_crop_size),
                    RandomHorizontalFlip(p=0.5),
                ]
                if self.training
                else [
                    ShortSideScale(self.video_min_short_side_scale),
                    CenterCrop(self.video_crop_size),
                ]
            ),
        )
        return Compose([video_transforms])

    def signal_transform(self, type: str = "spectrogram", sample_rate: int = 48000):
        if type == "spectrogram":
            n_fft = 1024
            win_length = None
            hop_length = 512

            transform = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
            )
        elif type == "melspectrogram":
            n_fft = 1024
            win_length = None
            hop_length = 512
            n_mels = 64

            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                onesided=True,
                n_mels=n_mels,
                mel_scale="htk",
            )
        elif type == "mfcc":
            n_fft = 2048
            win_length = None
            hop_length = 512
            n_mels = 256
            n_mfcc = 256

            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )
        else:
            raise ValueError(type)

        return transform

    def _preproc_audio(self, audio, audio_fps) -> Dict[str, Any]:
        # convert stero to mono
        # https://github.com/pytorch/audio/issues/363
        waveform_mono = torch.mean(audio, dim=0, keepdim=True)
        return {
            "signal": waveform_mono,
            "spectrogram": self.signal_transform(
                type=self.audio_transform_type,
                sample_rate=audio_fps,
            )(waveform_mono),
            "sampling_rate": audio_fps,
        }

    def convert_one_hot(self, label_list: List[str]) -> List[int]:
        labels = [x for x in label_list if x in self.label_name_id_map.keys()]
        assert len(labels) == len(
            label_list
        ), f"invalid filter {len(label_list)} -> {len(labels)}: {label_list}"
        one_hot = [0 for _ in range(self.num_classes)]
        for lab in labels:
            one_hot[self.label_name_id_map[lab]] = 1
        return one_hot
