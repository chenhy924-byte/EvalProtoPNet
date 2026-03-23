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
from pathlib import Path

import model, train_and_test as tnt
import util.utils as utils
from util.utils import str2bool
from torch.utils.tensorboard import SummaryWriter
from util.preprocess import mean, std
from util.eval_interpretability import evaluate_consistency
from util.datasets import Barefoot_Dataset

# https://github.com/hqhQAQ/EvalProtoPNet


def _barefoot_train_dir_and_nb_classes(data_path):
    """Resolve train folder and class count (same logic as Barefoot_Dataset / train.sh)."""
    train_dir = os.path.join(data_path, Barefoot_Dataset.TRAIN_DIR)
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(data_path, 'train_cropped')
    if not os.path.isdir(train_dir):
        return None, None
    nb = len(
        sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    )
    return train_dir, nb


def _fmt_lr_for_run_name(lr):
    """
    Match train.sh run folder strings: bash uses e.g. 1e-4, not 0.0001.
    """
    lr = float(lr)
    known = [
        (1e-4, '1e-4'),
        (3e-3, '3e-3'),
        (1e-6, '1e-6'),
        (1e-3, '1e-3'),
        (1e-2, '1e-2'),
    ]
    for val, s in known:
        if abs(lr - val) <= 1e-18 * max(1.0, abs(val)):
            return s
    return np.format_float_scientific(lr, precision=6, unique=True, trim='-')


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
    if args.eval: # Evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # Training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.base_architecture+ "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(
            tb_dir,
            args.base_architecture+ "_" + args.data_set
        ),
        flush_secs=1
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.base_architecture+ "_" + args.data_set + ".log"
        )
    )

    logger = logging.getLogger("main")
    return tb_writer, logger


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1028)
parser.add_argument('--output_dir', default='output_debug',
                    help='输出根目录。output_debug：自动追加时间戳子目录；'
                         '若恰为 output_cosine 且 Barefoot 训练，则自动变为 '
                         '<N>p_<arch>_YYYYMMDD_HHMMSS_<seed>_<lr>_<opt>_<epochs>_train（与 train.sh 一致）。')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--resume', default='', help='resume from checkpoint')  
# Data（赤足压力数据集：默认路径与子目录）
parser.add_argument('--data_set', default='Barefoot_Dataset', type=str)
parser.add_argument('--data_path', type=str, default='datasets/Barefoot_Dataset/')
parser.add_argument('--train_batch_size', default=80, type=int)
parser.add_argument('--test_batch_size', default=150, type=int)

# Model
parser.add_argument('--base_architecture', type=str, default='resnet34')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--save_ep_freq', default=400, type=int, help='save epoch frequency')
parser.add_argument('--num_prototypes_per_class', type=int, default=10, help='prototypes per class; total prototypes = num_classes * this')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=None, help='default: [num_classes*num_prototypes_per_class, 64, 1, 1]')
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

# Loss
parser.add_argument('--use_ortho_loss', type=str2bool, default=True)
parser.add_argument('--ortho_coe', type=float, default=1e-4)
parser.add_argument('--consis_coe', type=float, default=0.30)
parser.add_argument('--consis_thresh', type=float, default=0.10)

# Optimizer & Scheduler
parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER')
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
parser.add_argument('--features_lr', type=float, default=1e-4)
parser.add_argument('--add_on_layers_lr', type=float, default=3e-3)
parser.add_argument('--prototype_vectors_lr', type=float, default=3e-3)
parser.add_argument('--activation_weight_lr', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
parser.add_argument('--decay_epochs', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.1)

# Distributed training
parser.add_argument('--device', default=None, help='device to use (default: cuda if available else cpu)')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    args = parser.parse_args()

    # 先解析类别数（与 train.sh 一致），便于 output_cosine 下自动生成同名 run 目录
    train_dir = None
    if args.data_set == 'Barefoot_Dataset':
        train_dir, nb_early = _barefoot_train_dir_and_nb_classes(args.data_path)
        if train_dir is None:
            raise RuntimeError('Training directory not found under data_path: {}'.format(args.data_path))
        if nb_early == 0:
            raise RuntimeError('No class subdirectories in: {}'.format(train_dir))
        args.nb_classes = nb_early

    # 默认在 output_debug 下按时间戳创建子文件夹，避免覆盖；手动指定完整路径时则不追加时间戳
    out_base = args.output_dir.rstrip('/\\')
    if out_base == 'output_debug':
        args.output_dir = os.path.join('output_debug', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    elif (
        out_base == 'output_cosine'
        and args.data_set == 'Barefoot_Dataset'
        and not args.eval
    ):
        # 与 scripts/train.sh 一致: <N>p_<arch>_YYYYMMDD_HHMMSS_<seed>_<lr>_<opt>_<epochs>_train
        date_part = datetime.datetime.now().strftime('%Y%m%d')
        time_part = datetime.datetime.now().strftime('%H%M%S')
        lr_str = _fmt_lr_for_run_name(args.lr)
        run_name = (
            f"{args.nb_classes}p_{args.base_architecture}_{date_part}_{time_part}_"
            f"{args.seed}_{lr_str}_{args.opt}_{args.epochs}_train"
        )
        args.output_dir = os.path.join('output_cosine', run_name)

    # 设备自适应：统一使用变量 device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    __global_values__ = dict(it=0)
    seed = args.seed + utils.get_rank()
    set_seed(seed)

    # Distributed Training
    utils.init_distributed_mode(args)

    tb_writer, logger = get_outlog(args)

    # Setting Parameters
    base_architecture = args.base_architecture
    dataset_name = args.data_set

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
    model_dir = args.output_dir

    os.makedirs(model_dir, exist_ok=True)

    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    # 动态类别数：Barefoot 已在启动时解析；其它数据集保留原逻辑
    if args.data_set != 'Barefoot_Dataset':
        train_dir = os.path.join(args.data_path, Barefoot_Dataset.TRAIN_DIR)
        if not os.path.isdir(train_dir):
            train_dir = os.path.join(args.data_path, 'train_cropped')
        if not os.path.isdir(train_dir):
            raise RuntimeError('Training directory not found under data_path: {}'.format(args.data_path))
        args.nb_classes = len(
            sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        )
    # 原型层维度：M = num_classes * num_prototypes_per_class
    if args.prototype_shape is None:
        proto_dim = 64
        args.prototype_shape = [args.nb_classes * args.num_prototypes_per_class, proto_dim, 1, 1]
    img_size = args.input_size

    joint_optimizer_lrs = {'features': args.features_lr,
                        'add_on_layers': args.add_on_layers_lr,
                        'prototype_vectors': args.prototype_vectors_lr,
                        'activation_weight': args.activation_weight_lr}
    warm_optimizer_lrs = {'add_on_layers': args.add_on_layers_lr,
                        'prototype_vectors': args.prototype_vectors_lr,
                        'activation_weight': args.activation_weight_lr}
    coefs = {
        'crs_ent': 1,
        'orth': 1e-4,
        'clst': 0.8,
        'sep': -0.08,
        'consis': args.consis_coe,
    }

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 赤足压力数据集（损坏图片在初始化时跳过并记录日志）
    train_dataset = Barefoot_Dataset(
        args.data_path, train=True, transform=train_transform, check_integrity=True
    )
    test_dataset = Barefoot_Dataset(
        args.data_path, train=False, transform=test_transform, check_integrity=True
    )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(test_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    # train loader & test loader（多进程加载）
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.train_batch_size,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=args.test_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    # construct the model
    ppnet = model.construct_OursNet(base_architecture=args.base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=args.prototype_shape,
                                  num_classes=args.nb_classes,
                                  prototype_activation_function=args.prototype_activation_function,
                                  add_on_layers_type=args.add_on_layers_type)
    ppnet.to(device)
    ppnet_without_ddp = ppnet
    if args.distributed:
        ppnet = torch.nn.parallel.DistributedDataParallel(ppnet, device_ids=[args.gpu], find_unused_parameters=True)
        ppnet_without_ddp = ppnet.module
    n_parameters = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        ppnet_without_ddp.load_state_dict(checkpoint['model'])

    # Define optimizer
    joint_optimizer_specs = \
    [{'params': ppnet_without_ddp.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
     {'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet_without_ddp.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
     {'params': ppnet_without_ddp.activation_weight, 'lr': joint_optimizer_lrs['activation_weight']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)

    warm_optimizer_specs = \
    [{'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet_without_ddp.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
     {'params': ppnet_without_ddp.activation_weight, 'lr': warm_optimizer_lrs['activation_weight']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    max_accuracy, max_consis_score = 0.0, 0.0
    output_dir = Path(args.output_dir)

    # Train the model
    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            tnt.warm_only(model=ppnet)
            _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=warm_optimizer,
                        coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
        else:
            tnt.joint(model=ppnet)
            joint_lr_scheduler.step()
            _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=joint_optimizer,
                        coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])

        test_acc, losses = tnt.test(model=ppnet, epoch=epoch, dataloader=test_loader, coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
        tb_writer.add_scalar("epoch/val_acc1", test_acc, epoch)
        tb_writer.add_scalar("epoch/val_loss", losses['cross_entropy'], epoch)

        consistency_score = evaluate_consistency(ppnet, args)

        if utils.get_rank() == 0:
            logger.info(f"Consistency score of the network on the {len(test_dataset)} test images: {consistency_score:.2f}%")
            logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_acc:.2f}%")

        # Save best model by validation accuracy.
        if test_acc >= max_accuracy and utils.get_rank() == 0:
            best_path = output_dir / 'checkpoints/best_model.pth'
            utils.save_on_master({
                'model': ppnet_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
                'accuracy': float(test_acc),
                'consistency': float(consistency_score),
            }, best_path)

        # Keep final-epoch checkpoint as a separate file.
        if epoch == args.epochs - 1:
            final_path = output_dir / 'checkpoints/final_model.pth'
            utils.save_on_master({
                'model': ppnet_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
                'accuracy': float(test_acc),
                'consistency': float(consistency_score),
            }, final_path)
        max_accuracy = max(max_accuracy, test_acc)
        max_consis_score = max(max_consis_score, consistency_score)

        if utils.get_rank() == 0:
            logger.info(f'Max consistency score: {max_consis_score:.2f}%')
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # Disabled: no longer auto-generate accuracy_consistency_stability.txt.
