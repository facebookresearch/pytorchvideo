# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset


class AvaLabeledVideoFramePaths:
    """
    Pre-processor for the Ava Actions Dataset stored as image frames. 
    `<https://research.google.com/ava/download.html>_`
    This class handles the parsing of all the necessary CSV files containing
    frame paths and frame labels.

    Attributes:
        AVA_VALID_FRAMES (list): Range of valid annotated frames in Ava dataset.
        FPS (int): Frames per second in the dataset.
        AVA_VIDEO_START_SEC (int): Start time of the video in seconds.

    Class Methods:
        _aggregate_bboxes_labels(cls, inp: Dict):
            Aggregates bounding boxes and labels.

        from_csv(cls, frame_paths_file: str, frame_labels_file: str, video_path_prefix: str,
                 label_map_file: Optional[str] = None) -> AvaLabeledVideoFramePaths:
            Creates an instance of AvaLabeledVideoFramePaths from CSV files.

        load_and_parse_labels_csv(frame_labels_file: str, video_name_to_idx: dict,
                                  allowed_class_ids: Optional[Set] = None):
            Parses AVA per-frame labels from a CSV file.

        load_image_lists(frame_paths_file: str, video_path_prefix: str) -> Tuple:
            Loads image paths from a file and constructs dictionaries for video indexing.

        read_label_map(label_map_file: str) -> Tuple:
            Reads the label map and class IDs from a .pbtxt file.
    """

    # Range of valid annotated frames in Ava dataset
    AVA_VALID_FRAMES = list(range(902, 1799))
    FPS = 30
    AVA_VIDEO_START_SEC = 900

    @classmethod
    def _aggregate_bboxes_labels(cls, inp: Dict):

        # Needed for aggregating the bounding boxes
        labels = inp["labels"]
        extra_info = inp["extra_info"]
        boxes = inp["boxes"]

        labels_agg = []
        extra_info_agg = []
        boxes_agg = []
        bb_dict = {}

        for i in range(len(labels)):
            box_label, box_extra_info = labels[i], extra_info[i]

            bbox_key = "{:.2f},{:.2f},{:.2f},{:.2f}".format(
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            )

            if bbox_key not in bb_dict:
                bb_dict[bbox_key] = len(boxes_agg)
                boxes_agg.append(boxes[i])
                labels_agg.append([])
                extra_info_agg.append([])

            idx = bb_dict[bbox_key]
            labels_agg[idx].append(box_label)
            extra_info_agg[idx].append(box_extra_info)

        return {
            "labels": labels_agg,
            "boxes": boxes_agg,
            "extra_info": extra_info_agg,
        }

    @classmethod
    def from_csv(
        cls,
        frame_paths_file: str,
        frame_labels_file: str,
        video_path_prefix: str,
        label_map_file: Optional[str] = None,
    ) -> AvaLabeledVideoFramePaths:
        """
        Creates an AvaLabeledVideoFramePaths object from CSV files containing frame paths and labels.

        Args:
            frame_paths_file (str):
                Path to a file containing relative paths to all the frames in the video. Each line in the file
                is of the form <original_video_id video_id frame_id rel_path labels>.

            frame_labels_file (str):
                Path to the file containing labels per key frame. Acceptable file formats are as follows:
                Type 1 (CSV Columns):
                    - original_video_id
                    - frame_time_stamp
                    - bbox_x1
                    - bbox_y1
                    - bbox_x2
                    - bbox_y2
                    - action_label
                    - detection_iou

                Type 2 (CSV Columns):
                    - original_video_id
                    - frame_time_stamp
                    - bbox_x1
                    - bbox_y1
                    - bbox_x2
                    - bbox_y2
                    - action_label
                    - person_label

            video_path_prefix (str):
                Path to be augmented to each relative frame path to get the global frame path.

            label_map_file (str):
                Path to a .pbtxt file containing class IDs and class names. If not set, the label map is not
                loaded, and bbox labels are not pruned based on allowable class_ids in the label map.

        Returns:
            AvaLabeledVideoFramePaths: An AvaLabeledVideoFramePaths object.

        This class method initializes an AvaLabeledVideoFramePaths object from CSV files containing frame paths
        and labels. It processes these files to create a list of labeled video paths, where each entry is a tuple
        containing the path to the video frames directory and a label dictionary.

        Note: This function assumes specific CSV file formats and column names.
        """
        if label_map_file is not None:
            _, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(
                label_map_file
            )
        else:
            allowed_class_ids = None

        (
            image_paths,
            video_idx_to_name,
            video_name_to_idx,
        ) = AvaLabeledVideoFramePaths.load_image_lists(
            frame_paths_file, video_path_prefix
        )

        video_frame_labels = AvaLabeledVideoFramePaths.load_and_parse_labels_csv(
            frame_labels_file,
            video_name_to_idx,
            allowed_class_ids,
        )

        # Populate keyframes list
        labeled_video_paths = []
        for video_id in video_frame_labels.keys():
            for frame_video_sec in video_frame_labels[video_id].keys():
                labels = video_frame_labels[video_id][frame_video_sec]
                if len(labels["labels"]) > 0:
                    labels = AvaLabeledVideoFramePaths._aggregate_bboxes_labels(labels)
                    labels["video_index"] = video_id
                    labels["clip_index"] = frame_video_sec
                    video_frames_dir = os.path.dirname(image_paths[video_id][0])
                    labeled_video_paths.append((video_frames_dir, labels))

        return labeled_video_paths

    @staticmethod
    def load_and_parse_labels_csv(
        frame_labels_file: str,
        video_name_to_idx: dict,
        allowed_class_ids: Optional[Set] = None,
    ):
        """
        Parses AVA per-frame labels from a CSV file.

        Args:
            frame_labels_file (str):
                Path to the file containing labels per key frame. Acceptable file formats are as follows:
                Type 1 (CSV Columns):
                    - original_video_id
                    - frame_time_stamp
                    - bbox_x1
                    - bbox_y1
                    - bbox_x2
                    - bbox_y2
                    - action_label
                    - detection_iou

                Type 2 (CSV Columns):
                    - original_video_id
                    - frame_time_stamp
                    - bbox_x1
                    - bbox_y1
                    - bbox_x2
                    - bbox_y2
                    - action_label
                    - person_label

            video_name_to_idx (dict):
                A dictionary mapping video names to indices.

            allowed_class_ids (set):
                A set of unique integer class (bbox label) IDs that are allowed in the dataset.
                If not set, all class IDs are allowed in the bbox labels.

        Returns:
            dict: A dictionary containing labels for each keyframe in each video. The structure is as follows:
            {
                video_idx (int): {
                    frame_sec (float): {
                        'labels': List of bounding box labels,
                        'boxes': List of bounding boxes,
                        'extra_info': List of extra information containing either detections' IoU or person IDs
                    },
                    ...
                },
                ...
            }

        This function parses a CSV file containing per-frame labels, extracts the necessary information,
        and organizes it into a nested dictionary structure. The structure allows easy access to labels,
        bounding boxes, and extra information for each keyframe in each video.

        Note: This function assumes specific CSV file formats and column names.
        """
        labels_dict = {}
        with g_pathmgr.open(frame_labels_file, "r") as f:
            for line in f:
                row = line.strip().split(",")

                video_name = row[0]
                video_idx = video_name_to_idx[video_name]

                frame_sec = float(row[1])
                if (
                    frame_sec > AvaLabeledVideoFramePaths.AVA_VALID_FRAMES[-1]
                    or frame_sec < AvaLabeledVideoFramePaths.AVA_VALID_FRAMES[0]
                ):
                    continue

                # Since frame labels in video start from 0 not at 900 secs
                frame_sec = frame_sec - AvaLabeledVideoFramePaths.AVA_VIDEO_START_SEC

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                bbox = list(map(float, row[2:6]))

                # Label
                label = -1 if row[6] == "" else int(row[6])
                # Continue if the current label is not in allowed labels.
                if (allowed_class_ids is not None) and (label not in allowed_class_ids):
                    continue

                # Both id's and iou's are treated as float
                extra_info = float(row[7])

                if video_idx not in labels_dict:
                    labels_dict[video_idx] = {}

                if frame_sec not in labels_dict[video_idx]:
                    labels_dict[video_idx][frame_sec] = defaultdict(list)

                labels_dict[video_idx][frame_sec]["boxes"].append(bbox)
                labels_dict[video_idx][frame_sec]["labels"].append(label)
                labels_dict[video_idx][frame_sec]["extra_info"].append(extra_info)
        return labels_dict

    @staticmethod
    def load_image_lists(frame_paths_file: str, video_path_prefix: str) -> Tuple:
        """
        Loads image paths from the corresponding file.

        Args:
            frame_paths_file (str):
                Path to a file containing relative paths to all the frames in the video.
                Each line in the file is of the form <original_video_id video_id frame_id rel_path labels>.

            video_path_prefix (str):
                Path to be augmented to each relative frame path to get the global frame path.

        Returns:
            Tuple:
                A tuple containing the following elements:
                - image_paths_list (List[List[str]]): A list of lists containing absolute frame paths.
                The outer list is per video, and the inner list is per timestamp.
                - video_idx_to_name (Dict[int, str]): A dictionary mapping video index to video name.
                - video_name_to_idx (Dict[str, int]): A dictionary mapping video name to video index.

        This function parses a file containing frame paths and their associated video information.
        It organizes the frame paths into a list of lists, where each outer list represents a video,
        and each inner list represents timestamps within that video. The video information is also
        indexed and mapped for reference.

        The file format should follow:
        original_video_id video_id frame_id path labels
        """

        image_paths = []
        video_name_to_idx = {}
        video_idx_to_name = []

        with g_pathmgr.open(frame_paths_file, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)
                    image_paths.append({})

                data_key = video_name_to_idx[video_name]
                frame_id = int(row[2])
                image_paths[data_key][frame_id] = os.path.join(
                    video_path_prefix, row[3]
                )

        image_paths_list = []
        for i in range(len(image_paths)):
            image_paths_list.append([])
            sorted_keys = sorted(image_paths[i])
            for key in sorted_keys:
                image_paths_list[i].append(image_paths[i][key])

        return image_paths_list, video_idx_to_name, video_name_to_idx

    @staticmethod
    def read_label_map(label_map_file: str) -> Tuple:
        """
        Read a label map and extract class IDs and their associated class names.

        Args:
            label_map_file (str): The path to a .pbtxt file containing class IDs and class names.

        Returns:
            tuple: A tuple containing the following elements:
                - label_map (Dict[int, str]): A dictionary mapping class IDs (integers) to their associated class names (strings).
                - class_ids (Set[int]): A set containing unique class IDs (integers).

        This static method reads the contents of a .pbtxt file and extracts the class IDs and their associated class names.
        It returns a tuple containing a dictionary that maps class IDs to class names and a set of unique class IDs.
        """
        label_map = {}
        class_ids = set()
        name = ""
        class_id = ""
        with g_pathmgr.open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  name:"):
                    name = line.split('"')[1]
                elif line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    label_map[class_id] = name
                    class_ids.add(class_id)
        return label_map, class_ids


class TimeStampClipSampler:
    """
    A specialized clip sampler for sampling video clips around specific timestamps. This is particularly used
    in datasets like Ava where only a specific subset of clips in the video have annotations.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Initializes a TimeStampClipSampler.

        Args:
            clip_sampler (ClipSampler): The strategy used for sampling between the untrimmed clip boundary.
        """
        self.clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:
        """
        Samples a video clip around a specific timestamp.

        Args:
            last_clip_time (float): Not used for TimeStampClipSampler.
            video_duration (float): Not used for TimeStampClipSampler.
            annotation (Dict): A dictionary containing the time step to sample around.

        Returns:
            ClipInfo: An object including clip information with the following fields:
                - clip_start_time (float): The start time of the sampled clip in seconds.
                - clip_end_time (float): The end time of the sampled clip in seconds.
                - clip_index (int): Always 0.
                - aug_index (int): Always 0.
                - is_last_clip (bool): Always True.

        The `center_frame_sec` in the annotation dictionary represents the timestamp around which the clip is sampled.
        """
        center_frame_sec = annotation["clip_index"]  # a.k.a timestamp
        clip_start_sec = center_frame_sec - self.clip_sampler._clip_duration / 2.0
        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self.clip_sampler._clip_duration,
            0,
            0,
            True,
        )

    def reset(self) -> None:
        """
        Resets the TimeStampClipSampler.
        """
        pass



def Ava(
    frame_paths_file: str,
    frame_labels_file: str,
    video_path_prefix: str = "",
    label_map_file: Optional[str] = None,
    clip_sampler: Callable = ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
) -> None:
    """
    Creates a dataset for the AVA dataset with labeled video frames.

    Args:
        frame_paths_file (str): Path to a file containing relative paths
            to all the frames in the video. Each line in the file is of the
            form <original_video_id video_id frame_id rel_path labels>.
        frame_labels_file (str): Path to the file containing labels
            per key frame. Acceptable file formats are:
            Type 1:
                <original_video_id, frame_time_stamp, bbox_x_1, bbox_y_1, ...
                bbox_x_2, bbox_y_2, action_label, detection_iou>
            Type 2:
                <original_video_id, frame_time_stamp, bbox_x_1, bbox_y_1, ...
                bbox_x_2, bbox_y_2, action_label, person_label>.
        video_path_prefix (str): Path to be augmented to each relative frame
            path to obtain the global frame path.
        label_map_file (str): Path to a .pbtxt containing class IDs
            and class names. If not set, the label_map is not loaded, and bbox labels are
            not pruned based on allowable class_ids in label_map.
        clip_sampler (ClipSampler): Defines how clips should be sampled from each video.
        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
            video container. This defines the order in which videos are decoded and,
            if necessary, the distributed split.
        transform (Optional[Callable]): This callable is evaluated on the clip output
            and the corresponding bounding boxes before the clip and the bounding boxes
            are returned. It can be used for user-defined preprocessing and
            augmentations to the clips. If transform is None, the clip and bounding
            boxes are returned as they are.

    Returns:
        LabeledVideoDataset: A dataset containing labeled video frames for the AVA dataset.

    This function reads frame paths and labels from specified files, constructs a dataset, and returns it.
    """
    labeled_video_paths = AvaLabeledVideoFramePaths.from_csv(
        frame_paths_file,
        frame_labels_file,
        video_path_prefix,
        label_map_file,
    )
    return LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=TimeStampClipSampler(clip_sampler),
        transform=transform,
        video_sampler=video_sampler,
        decode_audio=False,
    )
