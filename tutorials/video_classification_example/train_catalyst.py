import argparse
import itertools
import logging
import os

from catalyst import dl, utils
import pytorchvideo.data
import pytorchvideo.models.resnet
import torch
import torch.nn.functional as F
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DistributedSampler, RandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


"""
This video classification example demonstrates how PyTorchVideo models, datasets and
transforms can be used with Catalyst. Specifically it shows how a
simple pipeline to train a Resnet on the Kinetics video dataset can be built.
"""

def _video_transform(args, mode: str):
    """
    This function contains example transforms using both PyTorchVideo and TorchVision
    in the same Callable. For 'train' mode, we use augmentations (prepended with
    'Random'), for 'valid' mode we use the respective determinstic function.
    """
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(args.video_num_subsampled),
                Normalize(args.video_means, args.video_stds),
            ]
            + (
                [
                    RandomShortSideScale(
                        min_size=args.video_min_short_side_scale,
                        max_size=args.video_max_short_side_scale,
                    ),
                    RandomCrop(args.video_crop_size),
                    RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                ]
                if mode == "train"
                else [
                    ShortSideScale(args.video_min_short_side_scale),
                    CenterCrop(args.video_crop_size),
                ]
            )
        ),
    )


def _audio_transform(args):
    """
    This function contains example transforms using both PyTorchVideo and TorchAudio
    in the same Callable.
    """
    n_fft = int(
        float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size
    )
    hop_length = int(
        float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size
    )
    eps = 1e-10
    return ApplyTransformToKey(
        key="audio",
        transform=Compose(
            [
                Resample(
                    orig_freq=args.audio_raw_sample_rate,
                    new_freq=args.audio_resampled_rate,
                ),
                MelSpectrogram(
                    sample_rate=args.audio_resampled_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=args.audio_num_mels,
                    center=False,
                ),
                Lambda(lambda x: x.clamp(min=eps)),
                Lambda(torch.log),
                UniformTemporalSubsample(args.audio_mel_num_subsample),
                Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                Lambda(
                    lambda x: x.view(1, x.size(0), 1, x.size(1))
                ),  # (T, F) -> (1, T, 1, F)
                Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
            ]
        ),
    )


def make_transforms(args, mode: str):
    """
    ##################
    # PTV Transforms #
    ##################

    # Each PyTorchVideo dataset has a "transform" arg. This arg takes a
    # Callable[[Dict], Any], and is used on the output Dict of the dataset to
    # define any application specific processing or augmentation. Transforms can
    # either be implemented by the user application or reused from any library
    # that's domain specific to the modality. E.g. for video we recommend using
    # TorchVision, for audio we recommend TorchAudio.
    #
    # To improve interoperation between domain transform libraries, PyTorchVideo
    # provides a dictionary transform API that provides:
    #   - ApplyTransformToKey(key, transform) - applies a transform to specific modality
    #   - RemoveKey(key) - remove a specific modality from the clip
    #
    # In the case that the recommended libraries don't provide transforms that
    # are common enough for PyTorchVideo use cases, PyTorchVideo will provide them in
    # the same structure as the recommended library. E.g. TorchVision didn't
    # have a RandomShortSideScale video transform so it's been added to PyTorchVideo.
    """
    if args.data_type == "video":
        transform = [
            _video_transform(args, mode),
            RemoveKey("audio"),
        ]
    elif args.data_type == "audio":
        transform = [
            _audio_transform(args),
            RemoveKey("video"),
        ]
    else:
        raise Exception(f"{args.data_type} not supported")

    return Compose(transform)


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def make_data(args):
    sampler = DistributedSampler if args.use_ddp else RandomSampler
    train_transform = make_transforms(args, mode="train")
    train_dataset = LimitDataset(
        pytorchvideo.data.Kinetics(
            data_path=os.path.join(args.data_path, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random", args.clip_duration
            ),
            video_path_prefix=args.video_path_prefix,
            transform=train_transform,
            video_sampler=sampler,
        )
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    sampler = DistributedSampler if args.use_ddp else None
    valid_transform = make_transforms(args, mode="valid")
    valid_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(args.data_path, "val.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler(
            "uniform", args.clip_duration
        ),
        video_path_prefix=args.video_path_prefix,
        transform=valid_transform,
        video_sampler=sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    return {"train": train_loader, "valid": valid_loader}


def make_components(args):
    if args.arch == "video_resnet":
        model = pytorchvideo.models.resnet.create_resnet(
            input_channel=3,
            model_num_class=400,
        )
        batch_key = "video"
    elif args.arch == "audio_resnet":
        model = pytorchvideo.models.resnet.create_acoustic_resnet(
            input_channel=1,
            model_num_class=400,
        )
        batch_key = "audio"
    else:
        raise Exception("{args.arch} not supported")

    criterion = torch.nn.CrossEntropy()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_epochs, last_epoch=-1
    )

    return model, criterion, optimizer, scheduler, batch_key



def train(args):
    # pytorch loaders
    loaders = make_data(args)
    # pytorch model, criterion, optimizer, scheduler... and some extras ;)
    model, criterion, optimizer, scheduler, batch_key = make_components(args)

    # model training
    """
    These keys are used in the inner loop of the training epoch as follows:

        batch[output_key] = model(batch[input_key])
        batch_metrics[loss_key] = criterion(batch[output_key], batch[target_key])
    
    PyTorchVideo batches are dictionaries containing each modality or metadata of
    the batch collated video clips. Kinetics contains the following notable keys:
       {
           'video': <video_tensor>,
           'audio': <audio_tensor>,
           'label': <action_label>,
       }
    - "video" is a Tensor of shape (batch, channels, time, height, Width)
    - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
    - "label" is a Tensor of shape (batch, 1)
    The PyTorchVideo models and transforms expect the same input shapes and
    dictionary structure making this function just a matter of unwrapping the dict and
    feeding it through the model/loss.
    """
    runner = dl.SupervisedRunner(
        input_key=batch_key, output_key="logits", target_key="label", loss_key="loss"
    )
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=args.logdir,
        num_epochs=args.max_epochs,
        valid_loader="valid",
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        verbose=False,
        ddp=args.use_ddp,
        fp16=args.use_fp16,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="label", topk_args=(1, 3)),
        ],
    )



def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


def main():
    """
    To train the ResNet with the Kinetics dataset we construct PyTorch modules above,
    and pass them to the train function of a catalyst.dl.SupervisedRunner.

    This example can be run either locally (with default parameters) or in distributed
    mode. To run with distributed backend provide the --use_ddp argument.
    """
    setup_logger()

    parser = argparse.ArgumentParser()

    #  Distributed parameters.
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")

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
    parser.add_argument("--max_epoch", default=200, type=int)
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

    parser.add_argument("--lodgir", default=None, type=str, required=True)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
