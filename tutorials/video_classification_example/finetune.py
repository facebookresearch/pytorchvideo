from pathlib import Path
from argparse import Namespace
from torchvision.transforms._transforms_video import CenterCropVideo
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import UniformClipSampler
import pytorch_lightning as pl
import torch
from pytorchvideo.models.head import create_res_basic_head
from torch import nn
from torch.optim import Adam

# HACK
from train import *


class UCF11DataModule(KineticsDataModule):

    def __init__(
        self,
        root="./",
        batch_size=32,
        num_workers=8,
        holdout_scene=None,
        side_size = 256,
        crop_size = 256,
        clip_mean = (0.45, 0.45, 0.45),
        clip_std = (0.225, 0.225, 0.225),
        num_frames = 8,
        sampling_rate = 8,
        frames_per_second = 30
    ):
        super().__init__(Namespace(data_type='video', batch_size=batch_size, workers=num_workers))

        self.root = Path(root) / 'action_youtube_naudio'
        assert self.root.exists(), "Dataset not found."
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.holdout_scene = holdout_scene
        self.side_size = side_size
        self.mean = clip_mean
        self.std = clip_std
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second
        self.clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second

        self.classes = [x.name for x in self.root.glob("*") if x.is_dir()]
        self.id_to_label = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_label = dict(zip(self.classes, range(len(self.classes))))
        self.num_classes = len(self.classes)


        # TODO - too many repeated .glob calls here.
        self.train_paths = []
        self.val_paths = []
        self.holdout_scenes = {}
        for c in self.classes:

            # Scenes within each class directory
            scene_names = sorted(set(x.name for x in (self.root / c).glob("*") if x.is_dir() and x.name != 'Annotation'))
            
            # Holdout the last scene
            # TODO - wrap this in a function so users can override the split logic
            holdout_scene = scene_names[-1]
            scene_names = scene_names[:-1]

            # Keep track of which scenes we held out for each class w/ a dict
            self.holdout_scenes[c] = holdout_scene

            # Prepare the list of 'labeled paths' required by the LabeledVideoDataset
            label_paths = [(v, {"label": self.class_to_label[c]}) for v in (self.root / c).glob("**/*.avi")]

            # HACK - this is no bueno. Can be done within the loop above
            self.train_paths.extend([x for x in label_paths if x[0].parent.name != holdout_scene])
            self.val_paths.extend([x for x in label_paths if x[0].parent.name == holdout_scene])

    def _video_transform(self, mode: str):
        # TODO - different tsfm for val/train
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.mean, self.std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
                ]
            ),
        )

    def _make_dataset(self, mode: str):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = DistributedSampler if (self.trainer is not None and self.trainer.use_ddp) else RandomSampler
        return LimitDataset(LabeledVideoDataset(
            self.train_paths if mode == 'train' else self.val_paths,
            UniformClipSampler(self.clip_duration),
            decode_audio=False,
            transform=self._make_transforms(mode=mode),
            video_sampler=sampler,
        ))

    def train_dataloader(self):
        self.train_dataset = self._make_dataset('train')
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        self.val_dataset = self._make_dataset('val')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class MiniKineticsDataModule(KineticsDataModule):
    TRAIN_PATH = 'train'
    VAL_PATH = 'val'


class Classifier(pl.LightningModule):

    def __init__(self, num_classes: int = 11, lr: float = 2e-4, freeze_backbone: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        resnet = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
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


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pytorch_lightning.Trainer.

    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

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

    # Data parameters.
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
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=200,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
        reload_dataloaders_every_epoch=False,
    )
    args = parser.parse_args()

    # Get data, model, configure trainer, and train
    data = MiniKineticsDataModule(args)
    model = Classifier(num_classes=6)
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=5)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
