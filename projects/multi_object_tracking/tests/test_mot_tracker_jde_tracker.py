import unittest

import torch
import os
from mot.tracker import JDETracker


def get_jde_tracker_inputs(directory, file_name):
    tracker_inputs = torch.load(os.path.join(directory, file_name))
    return tracker_inputs

class TestMOTTrackerJdeTracker(unittest.TestCase):
    def setUp(self):
        pass

    def test_jdetracker(self):
        tracker = JDETracker()
        for pred_dets, pred_embs in TestMOTTrackerJdeTracker._get_inputs():
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

