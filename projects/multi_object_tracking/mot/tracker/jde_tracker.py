import logging

import torch

from .base_jde import TrackState, BaseTrack, STrack
from .base_jde import joint_stracks, sub_stracks, remove_duplicate_stracks
from ..matching import jde_matching as matching
from ..motion import KalmanFilter


class JDETracker(object):
    """
    A simple and fast online association algorithm to work in conjuction with
    a "joint detection and embedding system.

    The original paper is:
    Towards Real-Time Multi-Object Tracking
    https://arxiv.org/abs/1909.12605

    The code from the author for this paper can be found at:
    https://github.com/Zhongdao/Towards-Realtime-MOT
    """

    def __init__(
            self,
            det_thresh: float = 0.3,
            track_buffer: int = 30,
            min_box_area: int = 200,
            tracked_thresh: float = 0.7,
            r_tracked_thresh: float = 0.5,
            unconfirmed_thresh: float = 0.7,
            motion: str = 'KalmanFilter',
            conf_thresh: int = 0,
            metric_type: str = 'euclidean'
    ) -> None:
        """
        Args:
            det_thresh (float): threshold of detection score
            track_buffer (int): buffer for tracker
            min_box_area (int): min box area to filter out low quality boxes
            tracked_thresh (float): linear assignment threshold of tracked
                stracks and detections
            r_tracked_thresh (float): linear assignment threshold of
                tracked stracks and unmatched detections
            unconfirmed_thresh (float): linear assignment threshold of
                unconfirmed stracks and unmatched detections
            motion (object): KalmanFilter instance
            conf_thres (float): confidence threshold for tracking
            metric_type (str): either "euclidean" or "cosine", the distance metric
                used for measurement to track association.
        """
        self.det_thresh = det_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.tracked_thresh = tracked_thresh
        self.r_tracked_thresh = r_tracked_thresh
        self.unconfirmed_thresh = unconfirmed_thresh

        if motion == 'KalmanFilter':
            self.motion = KalmanFilter()
        else:
            raise ValueError("Motion Filter incorrect/unavailable")

        self.conf_thresh = conf_thresh
        self.metric_type = metric_type

        self.frame_id = 0
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        # max_time_lost will be calculated: int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = 0


    def update(self,
               pred_dets: torch.Tensor,
               pred_embs: torch.Tensor):
        """
        Processes the detections(post NMS) and the embedding values.
        Associates the detection with corresponding tracklets and also handles
            lost, removed, refind and active tracklets.
        Note: that the detections and their corresponding embeddings need to be
        input only after being processed via techniques such as NMS. Also scale
        the detections to the original image size before passing to this method

        Args:
            pred_   dets (torch.Tensor): Detection results of the image, shape is [N, 5].
                                      i.e. (batch_id, x1, y1, x2, y2, object_conf)
            pred_embs (torch.Tensor): Embedding results of the image, shape is [N, 512]
                                      (as in the paper) or [N, M] where M = any sized feature
                                      embedding vector.

        Return:
            output_stracks (list): The list contains information regarding the
                online_tracklets for the recieved detection and embedding tensors
                of the corresponding image.
        """
        self.frame_id += 1
        activated_stracks = []  # for storing active tracks, for the current frame
        refind_stracks = []  # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []  # The tracks which are not obtained in the current frame but are not removed. (Lost for
        # some time lesser than the threshold for removing)
        removed_stracks = []

        if len(pred_dets) > 0:
            pred_dets = pred_dets.numpy()
            pred_embs = pred_embs.numpy()
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30)
                for (tlbrs, f) in zip(pred_dets, pred_embs)
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        """ Step 2: First association, with embedding"""
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with Kalman Filter
        STrack.multi_predict(strack_pool, self.motion)

        # The dists is the list of distances of the detection with the tracks in strack_pool
        dists = matching.embedding_distance(
            strack_pool, detections, metric=self.metric_type)
        dists = matching.fuse_motion(self.motion, dists, strack_pool,
                                     detections)

        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.tracked_thresh)

        # itracked is the id of the track and idet is the detection
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                # We have obtained a detection from a track which is not active,
                # hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        """ Step 3: Second association, with IOU"""
        # detections is now a list of the unmatched detections
        detections = [detections[i] for i in u_detection]

        # r_tracked_stracks is a container for stracks which were tracked till the previous
        # frame but no detection was found for it in the current frame.
        r_tracked_stracks = []

        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        # Same process done for some unmatched detections, but now considering IOU_distance as measure
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # matches is the list of detections which matched with corresponding
        # tracks by IOU distance method.
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.r_tracked_thresh)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list
        # and are marked lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.unconfirmed_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.motion, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated
        ]

        # TODO - Check whether we need to create such loggers
        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format(
        #     [track.track_id for track in activated_stracks]))
        # logger.debug('Refind: {}'.format(
        #     [track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format(
        #     [track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format(
        #     [track.track_id for track in removed_stracks]))

        return output_stracks
