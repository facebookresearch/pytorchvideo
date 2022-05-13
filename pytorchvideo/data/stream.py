
from typing import Any, Callable, Dict, Iterable
from .video import Video


class Stream(Iterable):
    """Create an iterable streaming clips of video data."""

    def __init__(
        self,
        video: Video,
        clip_duration: float,
        clip_transform: Callable = None,
        **get_clip_kwargs: Dict[str, Any],
    ) -> None:
        """
        Parameters
        ----------
        video : Video
            PyTorchVideo video instance to stream.
        clip_duration : float
            Maximum duration (in seconds) of the returned clip at every iteration.
        clip_transform : Transform, optional
            Optional transform to apply to each clip, by default None
        get_clip_kwargs : Dict[str, Any]
            Arguments to pass to the underlying video `get_clip` method.
        """
        super().__init__()
        self._clip_duration = clip_duration
        self._clip_transform = clip_transform
        self._video = video
        self._get_clip_kwargs = get_clip_kwargs

    def __iter__(self):
        current_time = 0.0
        while current_time < self._video.duration:
            next_time = min(
                self._video.duration,
                current_time + self._clip_duration,
            )
            video_data = self._video.get_clip(
                current_time,
                next_time,
                **self._get_clip_kwargs,
            )
            current_time = next_time

            if self._clip_transform:
                video_data = self._clip_transform(video_data)

            yield video_data

    @property
    def video(self):
        return self._video
