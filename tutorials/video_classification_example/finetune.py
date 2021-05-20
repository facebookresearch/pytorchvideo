import pytorch_lightning as pl

from data import UCF11DataModule
from models import SlowResnet50LightningModel
from train import parse_args


def train(args):
    pl.seed_everything(224)
    dm = UCF11DataModule(**vars(args))
    model = SlowResnet50LightningModel(num_classes=dm.NUM_CLASSES, **vars(args))
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


if __name__ == '__main__':
    main()
