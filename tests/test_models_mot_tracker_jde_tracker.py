import unittest

import torch
from pytorchvideo.models.mot.tracker import JDETracker
from utils import get_jde_tracker_inputs

class TestModelsMOTTrackerJdeTracker(unittest.TestCase):
    def setUp(self):
        pass

    def test_jdetracker(self):
        tracker = JDETracker()
        for pred_dets, pred_embs in TestModelsMOTTrackerJdeTracker._get_inputs():
            tracker.update(pred_dets, pred_embs)

    @staticmethod
    def _get_inputs():
        """

        Provide tensors(detection + feature embeddings) to the tracker as test cases.

        Yield:
            (torch.tensor, torch.tensor): tensor as test case input.
        """
        tracker_inputs = get_jde_tracker_inputs(directory=".", file_name="jde_dets.pt")
        for frame in tracker_inputs:
            yield tracker_inputs[frame][:, :5], tracker_inputs[frame][:, 6:]

