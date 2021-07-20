import numpy as np
from collections import OrderedDict
from numba import jit
from collections import deque

from ..motion.kalman_filter import KalmanFilter
from ..matching import jde_matching

# from utils.log import logger
# from models import *

class TrackState(object):
    """
    Class to store the TrackState
    Each BaseTrack class' (or its child class') object will have a TrackState defined

    Values are as follows:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    """
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    """
    Core class for storing and operating on a tracklet
    """
    def __init__(self,
                 tlwh: np.ndarray,
                 score: float,
                 temp_feat: np.ndarray,
                 buffer_size: int = 30
    ) -> None:
        """
        Args:
            tlwh (ndarray): Position of the bounding box obtained post detection
                            in the format`(top left x, top left y, width, height)`.
            score (float):
            temp_feat (ndarray): Embedding feature vector
            buffer_size (int): Maximum buffer size of the feature deque of the STrack
        """
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat: np.ndarray) -> None:
        """
        Update features of the STrack

        Args:
            feat(np.ndarray): Feature embedding vector of the new detection to be added to the STrack.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list,
                      kalman_filter: KalmanFilter):
        """
        Passes list of STracks through the Kalman filter to obtain the mean
        and covariance values of the predicted state

        Args:
            stracks(list): list of input STracks
            kalman_filter(KalmanFilter): KalmanFilter Object
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            #            multi_mean, multi_covariance = STrack.kalman_filter.multi_predict(multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self,
                 kalman_filter: KalmanFilter,
                 frame_id: int
    ) -> None:
        """
        Start a new tracklet

        Args:
            kalman_filter(KalmanFilter): For Motion prediction
            frame_id(int): The frame_id where the new STrack starts
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self,
                    new_track,
                    frame_id: int,
                    new_id: bool=False
    ) -> None:
        """
        Reactivate the unactive STrack

        Args:
            new_track: the new detection(STrack) to be added to the STrack
            frame_id(int): The frame id where this new detection was made
            new_id(bool): If true, updates the track_id of the STrack
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self,
               new_track,
               frame_id: int,
               update_feature: bool = True
    ) -> None:
        """
        Update a matched track

        Args:
            new_track: the new detection(STrack) to be added to the STrack
            frame_id(int): The frame id where this new detection was made
            update_feature(bool): If true, update the feature deque of this STrack.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    @jit
    def tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    @jit
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    # @staticmethod
    # @jit
    # def tlwh_to_tlbr(tlwh):
    #     ret = np.asarray(tlwh).copy()
    #     ret[2:] += ret[:2]
    #     return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def joint_stracks(tlista: list, tlistb: list) -> list:
    """
    Returns a union of the two input list STracks

    Args:
        tlista (list): first list of STracks
        tlistb (list): second list of STracks

    Returns:
        (list): Union of the two list of STracks
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: list, tlistb: list) -> list:
    """
    Subtracts a list of STracks from another i.e.
    subtracts 'tlistb' from 'tlista'

    Args:
        tlista (list): first list of STracks
        tlistb (list): second list of STracks

    Returns:
        (list): A list of track-id values obtained post subtraction.
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    Remove duplicate stracks

    Args:
        stracksa(list): first list of STracks
        stracksb(list): second list of STracks

    Returns:
        (list): first list of STracks with the duplicates removed.
        (list): second list of STracks with the duplicates removed.

    """
    pdist = jde_matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
