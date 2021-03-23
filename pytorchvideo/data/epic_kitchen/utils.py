# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict

from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.dataset_manifest_utils import EncodedVideoInfo, VideoFrameInfo
from pytorchvideo.data.utils import (
    optional_threaded_foreach,
    save_dataclass_objs_to_headered_csv,
)


def build_frame_manifest_from_flat_directory(
    data_directory_path: str, multithreaded: bool
) -> Dict[str, VideoFrameInfo]:
    """
    Args:
        data_directory_path (str): Path or URI to EpicKitchenDataset data.
                Data at this path must be a folder of structure:
                    {
                        "{video_id}": [
                            "frame_{frame_number}.{file_extension}",
                            "frame_{frame_number}.{file_extension}",
                            "frame_{frame_number}.{file_extension}",
                        ...]
                    ...}
        multithreaded (bool):
            controls whether io operations are performed across multiple threads.

    Returns:
        Dictionary mapping video_id of available videos to the locations of their
        underlying frame files.
    """

    video_frames = {}
    video_ids = g_pathmgr.ls(str(data_directory_path))

    def add_video_frames(video_id: str, video_path: str) -> None:
        video_frame_file_names = sorted(g_pathmgr.ls(video_path))
        for frame in video_frame_file_names:
            file_extension = frame.split(".")[-1]
            frame_name = frame[: -(len(file_extension) + 1)]
            stem, path_frame_id = frame_name.split("_")
            if video_id not in video_frames:
                video_frames[video_id] = VideoFrameInfo(
                    video_id=video_id,
                    location=video_path,
                    frame_file_stem=f"{stem}_",
                    frame_string_length=len(frame_name),
                    min_frame_number=int(path_frame_id),
                    max_frame_number=int(path_frame_id),
                    file_extension=file_extension,
                )
            else:
                video_frame_info = video_frames[video_id]
                # Check that this new frame is of the same format as other frames for this video
                # and that it is the next frame in order, if so update the frame info for this
                # video to reflect there is an additional frame.
                # We don't need to check video_id or frame_file_stem as they are function of
                # video_id which is aligned within the dictionary
                assert video_frame_info.frame_string_length == len(frame_name)
                assert video_frame_info.location == video_path, (
                    f"Frames for {video_id} found in two paths: "
                    f"{video_frame_info.location} and {video_path}"
                )
                assert video_frame_info.max_frame_number + 1 == int(path_frame_id)
                assert (
                    video_frame_info.file_extension == file_extension
                ), f"Frames with two different file extensions found for video {video_id}"
                video_frames[video_id] = VideoFrameInfo(
                    video_id=video_frame_info.video_id,
                    location=video_frame_info.location,
                    frame_file_stem=video_frame_info.frame_file_stem,
                    frame_string_length=video_frame_info.frame_string_length,
                    min_frame_number=video_frame_info.min_frame_number,
                    max_frame_number=int(path_frame_id),  # Update
                    file_extension=video_frame_info.file_extension,
                )

    video_paths = [
        (video_id, f"{data_directory_path}/{video_id}") for video_id in video_ids
    ]
    # Kick off frame indexing for all participants
    optional_threaded_foreach(add_video_frames, video_paths, multithreaded)

    return video_frames


def build_frame_manifest_from_nested_directory(
    data_directory_path: str, multithreaded: bool
) -> Dict[str, VideoFrameInfo]:
    """
    Args:
        data_directory_path (str): Path or URI to EpicKitchenDataset data.
            If this dataset is to load from the frame-based dataset:
                Data at this path must be a folder of structure:
    {
        "{participant_id}" : [
            "{participant_id}_{participant_video_id}_{frame_number}.{file_extension}",

        ...],
    ...}

        multithreaded (bool):
                controls whether io operations are performed across multiple threads.

        Returns:
            Dictionary mapping video_id of available videos to the locations of their
            underlying frame files.
    """

    participant_ids = g_pathmgr.ls(str(data_directory_path))
    video_frames = {}

    # Create function to execute in parallel that lists files available for each participant
    def add_participant_video_frames(
        participant_id: str, participant_path: str
    ) -> None:
        participant_frames = sorted(g_pathmgr.ls(str(participant_path)))
        for frame_file_name in participant_frames:
            file_extension = frame_file_name.split(".")[-1]
            frame_name = frame_file_name[: -(len(file_extension) + 1)]
            [path_participant_id, path_video_id, path_frame_id] = frame_name.split("_")
            assert path_participant_id == participant_id
            video_id = f"{path_participant_id}_{path_video_id}"
            if (
                video_id not in video_frames
            ):  # This is the first frame we have seen from video w/ video_id
                video_frames[video_id] = VideoFrameInfo(
                    video_id=video_id,
                    location=participant_path,
                    frame_file_stem=f"{video_id}_",
                    frame_string_length=len(frame_name),
                    min_frame_number=int(path_frame_id),
                    max_frame_number=int(path_frame_id),
                    file_extension=file_extension,
                )
            else:
                video_frame_info = video_frames[video_id]
                # Check that this new frame is of the same format as other frames for this video
                # and that it is the next frame in order, if so update the frame info for this
                # video to reflect there is an additional frame.
                # We don't need to check video_id or frame_file_stem as they are function of
                # video_id which is aligned within the dictionary
                assert video_frame_info.frame_string_length == len(frame_name)
                assert video_frame_info.location == participant_path, (
                    f"Frames for {video_id} found in two paths: "
                    f"{video_frame_info.location} and {participant_path}"
                )
                assert video_frame_info.max_frame_number + 1 == int(path_frame_id)
                assert (
                    video_frame_info.file_extension == file_extension
                ), f"Frames with two different file extensions found for video {video_id}"
                video_frames[video_id] = VideoFrameInfo(
                    video_id=video_frame_info.video_id,
                    location=video_frame_info.location,
                    frame_file_stem=video_frame_info.frame_file_stem,
                    frame_string_length=video_frame_info.frame_string_length,
                    min_frame_number=video_frame_info.min_frame_number,
                    max_frame_number=int(path_frame_id),  # Update
                    file_extension=video_frame_info.file_extension,
                )

    particpant_paths = [
        (participant_id, f"{data_directory_path}/{participant_id}")
        for participant_id in participant_ids
    ]
    # Kick off frame indexing for all participants
    optional_threaded_foreach(
        add_participant_video_frames, particpant_paths, multithreaded
    )

    return video_frames


def build_encoded_manifest_from_nested_directory(
    data_directory_path: str,
) -> Dict[str, EncodedVideoInfo]:
    """
    Creates a dictionary from video_id to EncodedVideoInfo for
    encoded videos in the given directory.

    Args:
        data_directory_path (str): The folder to ls to find encoded
        video files.

    Returns:
        Dict[str, EncodedVideoInfo] mapping video_id to EncodedVideoInfo
        for each file in 'data_directory_path'
    """
    encoded_video_infos = {}
    for participant_id in g_pathmgr.ls(data_directory_path):
        participant_folder_path = f"{data_directory_path}/{participant_id}"
        for video_file_name in g_pathmgr.ls(participant_folder_path):
            video_id = video_file_name[:6]
            video_full_path = f"{participant_folder_path}/{video_file_name}"
            encoded_video_infos[video_id] = EncodedVideoInfo(video_id, video_full_path)
    return encoded_video_infos
