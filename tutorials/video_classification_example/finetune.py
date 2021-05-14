from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from pytorchvideo.models.head import create_res_basic_head

from data import UCF11DataModule, KineticsDataModule, MiniKineticsDataModule
from models import Classifier


DATASET_MAP = {
    "ucf11": UCF11DataModule,
    "kinetics": KineticsDataModule,
    "kinetics-mini": MiniKineticsDataModule,
}


class Classifier(pl.LightningModule):

    def __init__(self, num_classes: int = 11, lr: float = 2e-4, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        resnet = torch.hub.load("facebookresearch/pytorchvideo", 'slow_r50', pretrained=self.hparams.pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[0][:-1])

        if self.hparams.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Head
        self.head = create_res_basic_head(in_features=2048, out_features=self.hparams.num_classes)

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.accuracy = {'train': self.train_acc, 'val': self.val_acc}

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        feats = self.backbone(x)
        return self.head(feats)

    def shared_step(self, batch, mode: str):
        y_hat = self(batch["video"])
        loss = self.loss_fn(y_hat, batch["label"])
        self.log(f"{mode}_loss", loss)

        if mode in ["val", "test"]:
            preds = y_hat.argmax(dim=1)
            acc = self.accuracy[mode](preds, batch["label"])
            self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def parse_args(args=None):
    parser = ArgumentParser()

    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="video_resnet",
        choices=["video_resnet", "audio_resnet"],
        type=str,
    )

    # Data parameters
    parser.add_argument(
        "--dataset", default="ucf11", choices=["ucf11", "kinetics", "kinetics-mini"]
    )
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=8, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_window_size", default=32, type=int)
    parser.add_argument("--audio_mel_step_size", default=16, type=int)
    parser.add_argument("--audio_num_mels", default=80, type=int)
    parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    # Trainer parameters.
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=200,
        callbacks=[pl.callbacks.LearningRateMonitor()],
        replace_sampler_ddp=False,
        reload_dataloaders_every_epoch=False,
    )
    return parser.parse_args(args=args)


def main(args):
    pl.trainer.seed_everything()
    dm_cls = DATASET_MAP.get(args.dataset)
    dm = dm_cls(args)
    model = Classifier(num_classes=dm_cls.NUM_CLASSES)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main(parse_args())
