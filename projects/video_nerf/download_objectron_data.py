# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

import requests


# URLs for downloading the Objectron dataset
public_url = "https://storage.googleapis.com/objectron"
blob_path = public_url + "/v1/index/chair_annotations_train"
video_ids = requests.get(blob_path).text
video_ids = video_ids.split("\n")

DATA_PATH = "./nerf/data/objectron"

os.makedirs(DATA_PATH, exist_ok=True)

# Download a video of a chair.
for i in range(3, 4):
    video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
    metadata_filename = public_url + "/videos/" + video_ids[i] + "/geometry.pbdata"
    annotation_filename = public_url + "/annotations/" + video_ids[i] + ".pbdata"

    # This file contains the bundle adjusted cameras
    sfm_filename = public_url + "/videos/" + video_ids[i] + "/sfm_arframe.pbdata"

    # video.content contains the video file.
    video = requests.get(video_filename)
    metadata = requests.get(metadata_filename)

    # Please refer to Parse Annotation tutorial to see how to parse the annotation files.
    annotation = requests.get(annotation_filename)

    sfm = requests.get(sfm_filename)

    video_path = os.path.join(DATA_PATH, "video.MOV")
    print("Writing video to %s" % video_path)
    file = open(video_path, "wb")
    file.write(video.content)
    file.close()

    geometry_path = os.path.join(DATA_PATH, "geometry.pbdata")
    print("Writing geometry data to %s" % geometry_path)
    file = open(geometry_path, "wb")
    file.write(metadata.content)
    file.close()

    annotation_path = os.path.join(DATA_PATH, "annotation.pbdata")
    print("Writing annotation data to %s" % annotation_path)
    file = open(annotation_path, "wb")
    file.write(annotation.content)
    file.close()

    sfm_arframe_path = os.path.join(DATA_PATH, "sfm_arframe.pbdata")
    print("Writing bundle adjusted camera data to %s" % sfm_arframe_path)
    file = open(sfm_arframe_path, "wb")
    file.write(sfm.content)
    file.close()
