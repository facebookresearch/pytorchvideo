import cv2
import torch


def process_video(video_path, model, max_frames):
    video = cv2.VideoCapture(video_path)

    frame_count = 0

    frame_tracker = []

    while True:
        is_frame, frame = video.read()

        if not is_frame:
            break

        # generate model output
        model_outputs = model.predict(frame)

        # get bounding boxes (x1, y1, x2, y2)
        bboxes_per_frame = model_outputs["instances"][
            model_outputs["instances"].pred_classes == 0
        ].pred_boxes
        bboxes_per_frame = bboxes_per_frame.tensor.to("cpu").squeeze()

        # calculate bbox center (x1, y1, x2, y2)
        bboxes_per_frame_center_x = (
            bboxes_per_frame[:, 0] + bboxes_per_frame[:, 2]
        ) / 2  # (x1+x2)/2
        bboxes_per_frame_center_y = (
            bboxes_per_frame[:, 1] + bboxes_per_frame[:, 3]
        ) / 2  # (y1+y2)/2

        # get keypoints
        keypoints_per_frame = model_outputs["instances"][
            model_outputs["instances"].pred_classes == 0
        ].pred_keypoints
        keypoints_per_frame = keypoints_per_frame[:, :, :2].to("cpu")

        # change origin of the keypoints to center of each bbox
        keypoints_per_frame[:, :, 0] = keypoints_per_frame[
            :, :, 0
        ] - bboxes_per_frame_center_x.unsqueeze(1)
        keypoints_per_frame[:, :, 1] = keypoints_per_frame[
            :, :, 1
        ] - bboxes_per_frame_center_y.unsqueeze(1)

        for i in range(bboxes_per_frame.shape[0]):
            bbox_coord = bboxes_per_frame[i, :]
            keypoint_per_bbox = keypoints_per_frame[i, :, :]

            bbox_info = {
                "bbox_id": i,
                "person_id": None,
                "frame_id": frame_count,
                "bbox_coord": bbox_coord,
                "keypoint_coord": keypoint_per_bbox,
            }

            frame_tracker.append(bbox_info)

        frame_count += 1

        if frame_count == max_frames:
            break

    video.release()
    cv2.destroyAllWindows()

    return frame_tracker


def create_keypoint_features_db(frame_tracker):
    return torch.stack([bbox["keypoint_coord"].flatten() for bbox in frame_tracker])


def calculate_distance(action_query, keypoint_db):
    scores = action_query @ keypoint_db.T
    return scores


def get_closest_matches(scores, method, n):
    if method == "topk":
        return torch.topk(scores, n).indices.squeeze()
    elif method == "softmax":
        score_probs = torch.nn.functional.softmax(scores, dim=1)
        return (score_probs > n).squeeze().nonzero()


## workflow

# frame_tracker = process_video(
#     video_path="./sample_video.mp4", model=keypoint_detection_model, max_frames=5
# )
#
# keypoint_db = create_keypoint_features_db(frame_tracker)
#
# scores = calculate_distance(action_query, keypoint_db)
#
# best_bbox_matches = get_closest_matches(scores, method="topk", n=10)
#
# for bbox_id in best_bbox_matches.tolist():
#   print(frame_tracker[bbox_id]["frame_id"])
