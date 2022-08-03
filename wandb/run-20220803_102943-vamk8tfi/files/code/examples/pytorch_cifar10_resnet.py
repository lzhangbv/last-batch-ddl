from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import wandb
#wandb = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import torch.multiprocessing as mp

import cifar_resnet as resnet
from cifar_wide_resnet import Wide_ResNet
from cifar_pyramidnet import ShakePyramidNet
from cifar_vgg import VGG
from vit import VisionTransformer, ViT_CONFIGS  
from efficientnet import EfficientNet
from utils import *

from opt import fpdp as hvd
import torch.distributed as dist

SPEED = False

def initialize():
    # Training Parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--model', type=str, default='resnet32',
                        help='ResNet model to use [20, 32, 56]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 or cifar100')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')

    # SGD Parameters
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-schedule', type=str, default='step', 
                        choices=['step', 'polynomial', 'cosine'], help='learning rate schedules')
    parser.add_argument('--lr-decay', nargs='+', type=float, default=[0.5, 0.75],
                        help='epoch intervals to decay lr when using step schedule')
    parser.add_argument('--lr-decay-alpha', type=float, default=0.1,
                        help='learning rate decay alpha')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--label-smoothing', type=int, default=0, metavar='WE',
                        help='label smoothing (default: 0)')
    parser.add_argument('--mixup', type=int, default=0, metavar='WE',
                        help='use mixup (default: 0)')
    parser.add_argument('--cutmix', type=int, default=0, metavar='WE',
                        help='use cutmix (default: 0)')
    parser.add_argument('--autoaugment', type=int, default=0, metavar='WE',
                        help='use autoaugment (default: 0)')
    parser.add_argument('--cutout', type=int, default=0, metavar='WE',
                        help='use cutout augment (default: 0)')
    parser.add_argument('--use-adam', type=int, default=0, metavar='WE',
                        help='use adam optimizer (default: 0)')
    parser.add_argument('--use-pretrained-model', type=int, default=0, metavar='WE',
                        help='use pretrained model e.g. ViT-B_16 (default: 0)')
    parser.add_argument('--pretrained-dir', type=str, default='/datasets/pretrained_models/',
                        help='pretrained model dir')

    # Last-batch Parameters
    parser.add_argument('--last-batch', type=int, default=1,
                        help='enable last batch optimizations')

    # Other Parameters
    parser.add_argument('--log-dir', default='./logs',
                        help='log directory')
    parser.add_argument('--dir', type=str, default='/datasets/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    
    # local_rank: (1) parse argument as follows in torch.distributed.launch; (2) read from environment in torch.distributed.run, i.e. local_rank=int(os.environ['LOCAL_RANK'])
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()


    # Training Settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.lr_decay[0] < 1.0: # epoch number percent
        args.lr_decay = [args.epochs * p for p in args.lr_decay]
    
    # Comm backend init
    # args.horovod = False

    dist.init_process_group(backend='nccl', init_method='env://')
    
    args.local_rank = int(os.environ['LOCAL_RANK'])
    logger.info("GPU %s out of %s GPUs", hvd.rank(), hvd.size())

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # Logging Settings
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir,
        '{}_{}_ep{}_bs{}_gpu{}_lb{}_{}_warmup{}_cutmix{}_aa{}.log'.format(args.dataset, args.model, args.epochs, args.batch_size, hvd.size(), args.last_batch, args.lr_schedule,args.warmup_epochs, args.cutmix, args.autoaugment))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    args.verbose = True if hvd.rank() == 0 else False
    
    if args.verbose:
        logger.info("torch version: %s", torch.__version__)
        logger.info(args)

    if args.verbose and wandb:
        wandb.init(project="last-batch-opt", entity="hkust-distributedml", name=logfile, config=args)
    
    return args


def get_dataset(args):
    # Load Cifar10
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    pretrained_model = args.model.lower() if args.use_pretrained_model else None
    transform_train, transform_test = get_transform(args.dataset, 
            args.autoaugment, args.cutout, pretrained_model)
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dir, train=True, 
                                     download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=args.dir, train=False,
                                    download=False, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=args.dir, train=True, 
                                     download=False, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=args.dir, train=False,
                                    download=False, transform=transform_test)


    # Use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    #train_loader = torch.utils.data.DataLoader(train_dataset,
    train_loader = MultiEpochsDataLoader(train_dataset,
            batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # Use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    #test_loader = torch.utils.data.DataLoader(test_dataset, 
    test_loader = MultiEpochsDataLoader(test_dataset, 
            batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)
    
    return train_sampler, train_loader, test_sampler, test_loader


def get_model(args):
    num_classes = 10 if args.dataset == 'cifar10' else 100
    # ResNet
    if args.model.lower() == "resnet20":
        model = resnet.resnet20(num_classes=num_classes)
    elif args.model.lower() == "resnet32":
        model = resnet.resnet32(num_classes=num_classes)
    elif args.model.lower() == "resnet44":
        model = resnet.resnet44(num_classes=num_classes)
    elif args.model.lower() == "resnet56":
        model = resnet.resnet56(num_classes=num_classes)
    elif args.model.lower() == "resnet110":
        model = resnet.resnet110(num_classes=num_classes)
    elif args.model.lower() == "wrn28-10":
        model = Wide_ResNet(28, 10, 0.0, num_classes=num_classes)
    elif args.model.lower() == "wrn28-20":
        model = Wide_ResNet(28, 20, 0.0, num_classes=num_classes)
    elif args.model.lower() == "pyramidnet":
        model = ShakePyramidNet(depth=110, alpha=270, num_classes=num_classes)
    elif args.model.lower() == "vgg16":
        model = VGG("VGG16", num_classes=num_classes)
    elif args.model.lower() == "vgg19":
        model = VGG("VGG19", num_classes=num_classes)
    elif args.model.lower() == "vit-b16" and args.use_pretrained_model:
        vit_config = ViT_CONFIGS[ "vit-b16"]
        model = VisionTransformer(vit_config, img_size=224, zero_head=True, num_classes=num_classes)
        model.load_from(np.load(args.pretrained_dir + "ViT-B_16.npz"))
    elif args.model.lower() == "vit-b16" and not args.use_pretrained_model:
        vit_config = ViT_CONFIGS[ "vit-b16"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif args.model.lower() == "vit-s8":
        vit_config = ViT_CONFIGS[ "vit-s8"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif args.model.lower() == "vit-t8":
        vit_config = ViT_CONFIGS[ "vit-t8"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif "efficientnet" in args.model.lower() and args.use_pretrained_model:
        model = EfficientNet.from_pretrained(args.model.lower(), 
                weights_path=args.pretrained_dir + args.model.lower() + ".pth",
                num_classes=num_classes)

    if args.cuda:
        model.cuda()

    # Optimizer
    if args.label_smoothing:
        criterion = LabelSmoothLoss(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    args.base_lr = args.base_lr * hvd.size()
    if args.use_adam:
        optimizer = optim.Adam(model.parameters(), 
                lr=args.base_lr, 
                betas=(0.9, 0.999), 
                weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), 
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    if args.last_batch:
        #optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), warmup_steps=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), warmup_steps=args.warmup_epochs * args.num_steps_per_epoch * 2)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Learning Rate Schedule
    if args.lr_schedule == 'cosine':
        lrs = create_cosine_lr_schedule(args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch)
    elif args.lr_schedule == 'polynomial':
        lrs = create_polynomial_lr_schedule(args.base_lr, args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch, lr_end=0.0, power=2.0)
    elif args.lr_schedule == 'step':
        lrs = create_multi_step_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay, args.lr_decay_alpha)
    
    lr_scheduler = LambdaLR(optimizer, lrs)

    return model, optimizer, lr_scheduler, criterion

def train(epoch, model, optimizer, lr_scheduler, criterion, train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)
    if args.cutmix:
        cutmix = CutMix(size=32, beta=1.0)
    elif args.mixup:
        mixup = MixUp(alpha=1.0)
    
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    avg_time = 0.0
    display = 10
    ittimes = []

    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        if args.cutmix:
            data, target, rand_target, lambda_ = cutmix((data, target))
        elif args.mixup:
            if np.random.rand() <= 0.8:
                data, target, rand_target, lambda_ = mixup((data, target))
            else:
                rand_target, lambda_ = torch.zeros_like(target), 1.0

        optimizer.zero_grad()

        output = model(data)
        if type(output) == tuple:
            output = output[0]

        if args.cutmix or args.mixup:
            loss = criterion(output, target) * lambda_ + criterion(output, rand_target) * (1.0 - lambda_)
        else:
            loss = criterion(output, target)
        
        with torch.no_grad():
            train_loss.update(loss)
            train_accuracy.update(accuracy(output, target))

        loss.backward()
        optimizer.step()
        avg_time += (time.time()-stime)
            
        if (batch_idx + 1) % display == 0:
            if args.verbose and SPEED:
                logger.info("[%d][%d] time: %.3f, speed: %.3f images/s" % (epoch, batch_idx, avg_time/display, args.batch_size/(avg_time/display)))
            ittimes.append(avg_time/display)
            avg_time = 0.0

        if batch_idx >= 60 and SPEED:
            if args.verbose:
                logger.info("Iteration time: mean %.3f, std: %.3f" % (np.mean(ittimes[1:]),np.std(ittimes[1:])))
            break

        if not args.lr_schedule == 'step':
            lr_scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))
        if wandb:
            wandb.log({"train loss": train_loss.avg.item(), "train acc": train_accuracy.avg.item()})

    if args.lr_schedule == 'step':
        lr_scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch learning rate: %f" % (epoch, optimizer.param_groups[0]['lr']))


def test(epoch, model, criterion, test_loader, args):
    model.eval()
    test_loss = Metric('val_loss')
    test_accuracy = Metric('val_accuracy')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if type(output) == tuple:
                output = output[0]
            test_loss.update(criterion(output, target))
            test_accuracy.update(accuracy(output, target))
            
    if args.verbose:
        logger.info("[%d] evaluation loss: %.4f, acc: %.3f" % (epoch, test_loss.avg.item(), 100*test_accuracy.avg.item()))
        if wandb:
            wandb.log({"eval loss": test_loss.avg.item(), "eval acc": test_accuracy.avg.item()})


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.set_start_method('forkserver')

    args = initialize()

    train_sampler, train_loader, _, test_loader = get_dataset(args)
    args.num_steps_per_epoch = len(train_loader)
    model, optimizer, lr_scheduler, criterion = get_model(args)

    start = time.time()

    for epoch in range(args.epochs):
        stime = time.time()
        train(epoch, model, optimizer, lr_scheduler, criterion, train_sampler, train_loader, args)
        if args.verbose:
            logger.info("[%d] epoch train time: %.3f"%(epoch, time.time() - stime))
        if not SPEED:
            test(epoch, model, criterion, test_loader, args)

    if args.verbose:
        logger.info("Total Training Time: %s", str(datetime.timedelta(seconds=time.time() - start)))

