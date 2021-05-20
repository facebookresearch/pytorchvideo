from argparse import ArgumentParser

import pytorch_lightning as pl
from data import LabeledVideoDataModule
from models import VideoClassificationLightningModule


def parse_args(args=None):
    parser = ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model Parameters.
    parser.add_argument("--lr", "--learning_rate", default=2e-4, type=float)

    # Data Parameters.
    parser = LabeledVideoDataModule.add_argparse_args(parser)

    # Training Parameters.
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        callbacks=[pl.callbacks.LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    return parser.parse_args(args)


def train(args):
    pl.seed_everything(224)
    dm = LabeledVideoDataModule.from_argparse_args(args)
    model = VideoClassificationLightningModule(num_classes=dm.NUM_CLASSES, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)


def main():
    args = parse_args()
    if args.on_cluster:
        from slurm import copy_and_run_with_config

        copy_and_run_with_config(
            train,
            args,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition=args.partition,
            gpus_per_node=args.gpus,
            ntasks_per_node=args.gpus,
            cpus_per_task=10,
            mem="470GB",
            nodes=args.num_nodes,
            constraint="volta32gb",
        )
    else:  # local
        train(args)


if __name__ == "__main__":
    main()
