#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pathlib
import shutil

import submitit


class InitAndRun(submitit.helpers.Checkpointable):
    def __init__(self, run_fn, run_config):
        self.run_fn = run_fn
        self.run_config = run_config
        
    def __call__(self):
        os.environ["RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        self.run_fn(self.run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    ignore_list = [
        "lightning_logs",
        "logs",
        "checkpoints",
        "experiments",
        ".git",
        "output",
        "val.csv",
        "train.csv",
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: ignore_list)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory,
                                      max_num_timeout=3)
    executor.update_parameters(**cluster_config)
    job = executor.submit(InitAndRun(run_fn, run_config))
    print(f"job_id: {job}")
