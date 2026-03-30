# python train_baseline.py --model_name resnet50 --data_path /root/autodl-tmp/datasets/Barefoot_Dataset/ --epochs 30 && \
# python train_baseline.py --model_name vgg16_bn --data_path /root/autodl-tmp/datasets/Barefoot_Dataset/ --epochs 30 && \
# python train_baseline.py --model_name densenet121 --data_path /root/autodl-tmp/datasets/Barefoot_Dataset/ --epochs 30  
# python train_baseline.py --data_path datasets/Barefoot_Dataset_2 --model_name resnet50 --pretrained False --epochs 2
# python train_baseline.py --data_path datasets/Barefoot_Dataset_2 --model_name vgg16 --pretrained False --epochs 2
# python train_baseline.py --data_path datasets/Barefoot_Dataset_2 --model_name densenet121 --pretrained False --epochs 2
import os
import re
import time
import torch
import shutil
import random
import logging
import datetime
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as tv_models
from pathlib import Path
from contextlib import nullcontext
from urllib.parse import urlparse

import util.utils as utils
from util.utils import str2bool
from torch.utils.tensorboard import SummaryWriter
from util.preprocess import mean, std
from util.datasets import Barefoot_Dataset

def _barefoot_train_dir_and_nb_classes(data_path):
    train_dir = os.path.join(data_path, Barefoot_Dataset.TRAIN_DIR)
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(data_path, "train_cropped")
    if not os.path.isdir(train_dir):
        return None, None
    nb = len(sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]))
    return train_dir, nb


def _fmt_lr_for_run_name(lr):
    lr = float(lr)
    known = [
        (1e-4, "1e-4"),
        (3e-3, "3e-3"),
        (1e-6, "1e-6"),
        (1e-3, "1e-3"),
        (1e-2, "1e-2"),
    ]
    for val, s in known:
        if abs(lr - val) <= 1e-18 * max(1.0, abs(val)):
            return s
    return np.format_float_scientific(lr, precision=6, unique=True, trim="-")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_outlog(args):
    if args.eval:
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else:
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.model_name + "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(tb_dir, args.model_name + "_" + args.data_set),
        flush_secs=1,
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(logfile_dir, args.model_name + "_" + args.data_set + ".log"),
    )
    logger = logging.getLogger("train_baseline")
    return tb_writer, logger


def _build_torchvision_model(model_name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        m = tv_models.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = torch.nn.Linear(in_features, num_classes)
        return m
    if model_name == "vgg16":
        if pretrained:
            weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        else:
            weights = None
        m = tv_models.vgg16(weights=weights)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return m
    if model_name == "densenet121":
        if pretrained:
            weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None
        m = tv_models.densenet121(weights=weights)
        in_features = m.classifier.in_features
        m.classifier = torch.nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unsupported --model_name: {model_name}")


def _delete_torch_hub_checkpoint_from_url(url: str) -> None:
    # torch.hub uses ~/.cache/torch/hub/checkpoints/<filename>
    try:
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            return
        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        ckpt_path = os.path.join(hub_dir, filename)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    except Exception:
        return


def build_model_with_retry(model_name: str, num_classes: int, pretrained: bool, max_retries: int = 3) -> torch.nn.Module:
    """
    Work around corrupted torchvision weight downloads (e.g. unexpected EOF / central directory missing).
    """
    last_err = None
    for _ in range(max_retries):
        try:
            return _build_torchvision_model(model_name, num_classes, pretrained=pretrained)
        except RuntimeError as e:
            msg = str(e).lower()
            if any(k in msg for k in ["central directory", "unexpected eof", "file might be corrupted"]):
                # delete cached weight and retry
                if model_name.lower() == "resnet50":
                    _delete_torch_hub_checkpoint_from_url(tv_models.ResNet50_Weights.IMAGENET1K_V1.url)
                elif model_name.lower() == "vgg16":
                    _delete_torch_hub_checkpoint_from_url(tv_models.VGG16_Weights.IMAGENET1K_V1.url)
                elif model_name.lower() == "densenet121":
                    _delete_torch_hub_checkpoint_from_url(tv_models.DenseNet121_Weights.IMAGENET1K_V1.url)
                last_err = e
                continue
            raise
        except Exception as e:
            last_err = e
            break
    if last_err is not None:
        raise last_err


def _amp_context(args, device):
    if not getattr(args, "use_amp", False):
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    if getattr(args, "use_bf16", False):
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if getattr(args, "use_fp16", False):
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, dataloader, optimizer, epoch, tb_writer, iteration, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 60
    dev = args.device
    scaler = getattr(args, "_scaler", None)

    n_examples, n_correct, n_batches = 0, 0, 0
    total_loss = 0.0

    for images, labels in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)

        optimizer.zero_grad()
        with _amp_context(args, dev):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

        if args.use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # stats
        n_batches += 1
        total_loss += float(loss.item())
        _, pred = torch.max(logits.data, 1)
        n_examples += labels.size(0)
        n_correct += (pred == labels).sum().item()

        metric_logger.update(loss=float(loss.item()))
        if utils.get_rank() == 0:
            tb_writer.add_scalars(
                main_tag="train/loss",
                tag_scalar_dict={"cls": float(loss.item())},
                global_step=iteration + n_batches,
            )

    acc = (n_correct / max(1, n_examples)) * 100.0
    avg_loss = total_loss / max(1, n_batches)
    return acc, {"cross_entropy": avg_loss}


@torch.no_grad()
def evaluate(model, dataloader, epoch, tb_writer, iteration, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: [{epoch}]"
    print_freq = 60
    dev = args.device

    n_examples, n_correct, n_batches = 0, 0, 0
    total_loss = 0.0

    for images, labels in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)

        with _amp_context(args, dev):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

        n_batches += 1
        total_loss += float(loss.item())
        _, pred = torch.max(logits.data, 1)
        n_examples += labels.size(0)
        n_correct += (pred == labels).sum().item()

        metric_logger.update(loss=float(loss.item()))
        if utils.get_rank() == 0:
            tb_writer.add_scalars(
                main_tag="test/loss",
                tag_scalar_dict={"cls": float(loss.item())},
                global_step=iteration + n_batches,
            )

    acc = (n_correct / max(1, n_examples)) * 100.0
    avg_loss = total_loss / max(1, n_batches)
    return acc, {"cross_entropy": avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument(
        "--output_dir",
        default="output_cosine",
        help="Output root. If equals output_cosine, auto-create run subdir like main.py.",
    )
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    # Data
    parser.add_argument("--data_set", default="Barefoot_Dataset", type=str)
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/Barefoot_Dataset/",
        help="Barefoot root (train_cropped_augmented / test_cropped); trailing slash optional, matching train.sh",
    )
    # Match scripts/train.sh (paper-aligned run) — not main.py argparse defaults, which train.sh overrides.
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--input_size", default=224, type=int)

    # Baseline model
    parser.add_argument("--model_name", default="resnet50", choices=["resnet50", "vgg16", "densenet121"])
    parser.add_argument("--pretrained", type=str2bool, default=True)

    # Optimizer & Scheduler (keep consistent knobs)
    parser.add_argument("--opt", default="adam", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    # StepLR after warmup — same step_size/gamma as scripts/train.sh → main.py joint_lr_scheduler
    parser.add_argument("--decay_epochs", type=int, default=3)
    parser.add_argument("--decay_rate", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Adam weight decay; main uses 1e-3 on tuned param groups")

    # Device / distributed
    parser.add_argument("--device", default=None)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist-eval", action="store_true", default=False)

    # AMP
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--auto_device", type=str2bool, default=True, help="auto-select cpu/cuda and multi-gpu behavior")

    args = parser.parse_args()
    # Align path handling with scripts/train.sh (data_path="${data_path%/}")
    args.data_path = os.path.normpath(args.data_path.rstrip("/\\"))

    # device / gpu count behavior (similar to main.py expectations)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.device is not None:
        device = torch.device(args.device)
    elif args.auto_device:
        device = torch.device("cuda" if cuda_count > 0 else "cpu")
    else:
        device = torch.device("cpu")
    args.device = device

    # class count
    if args.data_set != "Barefoot_Dataset":
        raise RuntimeError("train_baseline.py currently supports Barefoot_Dataset only.")
    train_dir, nb = _barefoot_train_dir_and_nb_classes(args.data_path)
    if train_dir is None or nb == 0:
        raise RuntimeError(f"Training directory not found or empty under data_path: {args.data_path}")
    args.nb_classes = nb

    # output_dir naming (align style with main.py output_cosine)
    out_base = args.output_dir.rstrip("/\\")
    if out_base == "output_cosine" and not args.eval:
        date_part = datetime.datetime.now().strftime("%Y%m%d")
        time_part = datetime.datetime.now().strftime("%H%M%S")
        lr_str = _fmt_lr_for_run_name(args.lr)
        run_name = f"{args.nb_classes}p_{args.model_name}_{date_part}_{time_part}_{args.seed}_{lr_str}_{args.opt}_{args.epochs}_train"
        args.output_dir = os.path.join("output_cosine", run_name)

    # distributed / multi-gpu behavior:
    # - no gpu: cpu single process
    # - 1 gpu: single process (no DDP)
    # - >=2 gpus: use DDP when launched with torchrun, otherwise fallback to DataParallel
    utils.init_distributed_mode(args)

    # seed
    seed = args.seed + utils.get_rank()
    set_seed(seed)

    # AMP setup
    bf16_supported = False
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = torch.cuda.is_bf16_supported()
        else:
            major, _ = torch.cuda.get_device_capability()
            bf16_supported = major >= 8
    args.use_amp = bool(args.use_amp and device.type == "cuda")
    args.use_bf16 = bool(args.use_amp and args.amp_dtype == "bf16" and bf16_supported)
    args.use_fp16 = bool(args.use_amp and args.amp_dtype == "fp16")
    if args.amp_dtype == "bf16" and args.use_amp and device.type == "cuda" and not bf16_supported:
        print("Warning: BF16 not supported on current CUDA device, fallback to FP32.")
        args.use_bf16 = False
        args.use_fp16 = False
    if args.use_fp16:
        args._scaler = torch.cuda.amp.GradScaler()

    tb_writer, logger = get_outlog(args)
    logger.info(
        f"Baseline={args.model_name}, classes={args.nb_classes}, "
        f"AMP enabled={args.use_amp}, dtype={args.amp_dtype}, "
        f"use_bf16={args.use_bf16}, device={args.device}"
    )

    # transforms & datasets (reuse same preprocessing as main.py)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(args.input_size, args.input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size=(args.input_size, args.input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = Barefoot_Dataset(args.data_path, train=True, transform=train_transform, check_integrity=True)
    test_dataset = Barefoot_Dataset(args.data_path, train=False, transform=test_transform, check_integrity=True)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.train_batch_size,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_val,
        batch_size=args.test_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    # model (NO prototype/SDFA modules)
    model = build_model_with_retry(args.model_name, args.nb_classes, pretrained=args.pretrained).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    elif device.type == "cuda" and cuda_count >= 2:
        logger.info(f"Multiple GPUs detected (count={cuda_count}) but not launched via torchrun; using DataParallel.")
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    # Optimizer & StepLR: mirror main.py joint_optimizer + joint_lr_scheduler after warmup (train.sh passes these hparams).
    if args.opt.lower() != "adam":
        raise RuntimeError("Only adam is supported in train_baseline.py to match main.py / train.sh defaults.")
    optimizer = torch.optim.Adam(
        model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_epochs, gamma=args.decay_rate
    )

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # keep code snapshot for reproducibility
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=output_dir)

    best_acc = 0.0
    best_epoch = -1
    start_time = time.time()
    iteration = 0
    for epoch in range(args.epochs):
        if args.distributed and hasattr(sampler_train, "set_epoch"):
            sampler_train.set_epoch(epoch)

        if epoch >= args.warmup_epochs:
            lr_scheduler.step()

        train_acc, _ = train_one_epoch(model, train_loader, optimizer, epoch, tb_writer, iteration, args)
        test_acc, losses = evaluate(model, test_loader, epoch, tb_writer, iteration, args)
        if utils.get_rank() == 0:
            tb_writer.add_scalar("epoch/val_acc1", test_acc, epoch)
            tb_writer.add_scalar("epoch/val_loss", losses["cross_entropy"], epoch)
            logger.info(f"Accuracy on {len(test_dataset)} test images: {test_acc:.2f}%")

        # checkpoints: best + final (per model_name)
        if test_acc >= best_acc and utils.get_rank() == 0:
            best_acc = float(test_acc)
            best_epoch = int(epoch)
            best_path = output_dir / f"checkpoints/best_{args.model_name}.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "accuracy": float(test_acc),
                },
                best_path,
            )

        if epoch == args.epochs - 1 and utils.get_rank() == 0:
            final_path = output_dir / f"checkpoints/final_{args.model_name}.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "accuracy": float(test_acc),
                },
                final_path,
            )

    total_time = time.time() - start_time
    logger.info("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    if utils.get_rank() == 0:
        logger.info(f"Max accuracy: {best_acc:.2f}% (epoch={best_epoch})")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

