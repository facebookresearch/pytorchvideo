from collections import OrderedDict
from typing import Callable

import torch
from hook import HookBase


def create_keypoint_features_db(frame_tracker):
    return torch.stack([bbox["keypoint_coord"].flatten() for bbox in frame_tracker])


def calculate_distance_scores(action_query, keypoint_feature_db):
    scores = action_query @ keypoint_feature_db.T
    return scores


def get_closest_keypoint_feature_match(scores, method, n):
    if method == "topk":
        return torch.topk(scores, n).indices.squeeze().tolist()
    elif method == "softmax":
        score_probs = torch.nn.functional.softmax(scores, dim=1)
        return (score_probs > n).squeeze().nonzero().tolist()[0]


def bbox_to_frame_executor(frame_tracker, best_bbox_matches):
    return [frame_tracker[bbox_id]["frame_id"] for bbox_id in best_bbox_matches]


class PeopleKeypointRetrievalHook(HookBase):
    def __init__(self, executor: Callable = bbox_to_frame_executor):
        self.executor = executor
        self.inputs = ["frame_tracker", "action_query"]
        self.outputs = ["frame_id"]

    def _run(self, status: OrderedDict):
        # extract frame_tracker and action_query feature
        frame_tracker = status["frame_tracker"]
        action_query = status["action_query"]

        # combine multiple keypoint features into a single tensor
        keypoint_feature_db = create_keypoint_features_db(frame_tracker)

        # find feature closest to action_query from the keypoint_feature_db
        distance_scores = calculate_distance_scores(
            action_query=action_query, keypoint_feature_db=keypoint_feature_db
        )

        # extract the index (bbox_id) of the best matches
        best_bbox_match_list = get_closest_keypoint_feature_match(
            scores=distance_scores, method="softmax", n=0.9
        )

        # get frame_id_list from the best_bbox_match_list
        frame_id_list = self.executor(
            frame_tracker=frame_tracker, best_bbox_matches=best_bbox_match_list
        )

        return {"frame_id_list": frame_id_list}
