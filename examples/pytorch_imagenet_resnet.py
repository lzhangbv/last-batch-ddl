from __future__ import print_function

import time
from datetime import datetime, timedelta
import argparse
import os
import math
import sys
import warnings
import numpy as np
from distutils.version import LooseVersion
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
strhdlr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

SPEED = False

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision import datasets, transforms
#import horovod.torch as hvd
from opt import fpdp as hvd
import torch.distributed as dist
from tqdm import tqdm
from distutils.version import LooseVersion
import imagenet_resnet as models
import imagenet_inceptionv4 as inceptionv4
from utils import *

#os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '10' 

STEP_FIRST = LooseVersion(torch.__version__) < LooseVersion('1.1.0')

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def initialize():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--horovod', type=int, default=1, metavar='N',
                        help='whether use horovod as communication backend (default: 1)')
    parser.add_argument('--train-dir', default='/tmp/imagenet/ILSVRC2012_img_train/',
                        help='path to training data')
    parser.add_argument('--val-dir', default='/tmp/imagenet/ILSVRC2012_img_val/',
                        help='path to validation data')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--model', default='resnet50',
                        help='Model (resnet35, resnet50, resnet101, resnet152, resnext50, resnext101)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lr-schedule', type=str, default='step', 
                        choices=['step', 'polynomial', 'cosine'], help='learning rate schedules')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[30, 60, 80],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing (default 0.1)')
    
    # Last-batch Parameters
    parser.add_argument('--last-batch', type=int, default=1,
                        help='enable last batch optimizations')
    parser.add_argument('--sync-warmup', type=int, default=0,
                        help='enable synchronous warmup')
    parser.add_argument('--switch-decay', type=int, default=0,
                        help='enable switch decay')
    
    # Other Parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--single-threaded', action='store_true', default=False,
                        help='disables multi-threaded dataloading')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Comm backend init
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.manual_seed(args.seed)
    args.verbose = 1 if hvd.rank() == 0 else 0
    #if args.verbose:
    #    logger.info(args)

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    args.log_dir = os.path.join(args.log_dir, 
                                "imagenet_resnet50_gpu{}_lb{}_{}".format(
                                hvd.size(), args.last_batch, 
                                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.checkpoint_format=os.path.join(args.log_dir, args.checkpoint_format)
    #os.makedirs(args.log_dir, exist_ok=True)

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    logfile = './logs/imagenet_{}_gpu{}_bs{}_ep{}_lb{}.log'.format(args.model, hvd.size(), args.batch_size, args.epochs, args.last_batch)
    
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    if args.verbose:
        logger.info(args)

    return args

def get_datasets(args):
    # Horovod: limit # of CPU threads to be used per worker.
    if args.single_threaded:
        torch.set_num_threads(4)
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    else:
        torch.set_num_threads(8)
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
    val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size * args.batches_per_allreduce,
            sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size,
            sampler=val_sampler, **kwargs)

    return train_sampler, train_loader, val_sampler, val_loader

def get_model(args):
    if args.model.lower() == 'resnet34':
        model = models.resnet34()
    elif args.model.lower() == 'resnet50':
        model = models.resnet50()
    elif args.model.lower() == 'resnet101':
        model = models.resnet101()
    elif args.model.lower() == 'resnet152':
        model = models.resnet152()
    elif args.model.lower() == 'resnext50':
        model = models.resnext50_32x4d()
    elif args.model.lower() == 'resnext101':
        model = models.resnext101_32x8d()
    elif args.model.lower() == 'densenet121':
        model = torchvision.models.densenet121(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'densenet201':
        model = torchvision.models.densenet201(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'vgg16':
        model = torchvision.models.vgg16(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'inceptionv3':
        model = torchvision.models.inception_v3(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'inceptionv4':
        model = inceptionv4.inceptionv4(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2()
    else:
        raise ValueError('Unknown model \'{}\''.format(args.model))

    if args.cuda:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    args.base_lr = args.base_lr * hvd.size() * args.batches_per_allreduce
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=args.momentum, weight_decay=args.wd)
    #optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-09)

    if args.last_batch:
        warmup_steps = args.warmup_epochs * args.num_steps_per_epoch if args.sync_warmup else 0
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), warmup_steps=warmup_steps)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights 
    # to other workers.
    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Learning Rate Schedule
    if args.lr_schedule == 'cosine':
        lrs = create_cosine_lr_schedule(args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch)
        if args.switch_decay: 
            lrs = create_cosine_lr_schedule_with_decay(args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch)
    elif args.lr_schedule == 'polynomial':
        lrs = create_polynomial_lr_schedule(args.base_lr, args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch, lr_end=0.0, power=2.0)
    elif args.lr_schedule == 'step':
        lrs = create_multi_step_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay, args.lr_decay_alpha)
    
    lr_scheduler = LambdaLR(optimizer, lrs)
    loss_func = LabelSmoothLoss(args.label_smoothing)

    return model, optimizer, lr_scheduler, loss_func

def train(epoch, model, optimizer, lr_scheduler,
          loss_func, train_sampler, train_loader, args):

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    avg_time = 0.0
    ittimes = []
    display=20

    profiling=True
    if True:
        for batch_idx, (data, target) in enumerate(train_loader):
            stime = time.time()
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)

                loss = loss_func(output, target_batch)

                with torch.no_grad():
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(output, target_batch))

                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()        

            optimizer.step()
            avg_time += (time.time()-stime)

            if batch_idx > 0 and batch_idx % display == 0:
                if args.verbose:
                    logger.info("[%d][%d] loss: %.4f, acc: %.2f, time: %.3f" % (epoch, batch_idx, train_loss.avg.item(), 100*train_accuracy.avg.item(), avg_time/display))
                ittimes.append(avg_time/display)
                avg_time = 0.0
            if batch_idx > 120 and SPEED:
                if args.verbose:
                    logger.info("Iteration time: mean %.3f, std: %.3f" % (np.mean(ittimes[1:]),np.std(ittimes[1:])))
                break
        
            if not args.lr_schedule == 'step':
                lr_scheduler.step()
        
        if args.verbose:
            logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))

    
        if args.lr_schedule == 'step':
            lr_scheduler.step()


def validate(epoch, model, loss_func, val_loader, args):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    if True:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))
            if args.verbose:
                logger.info("[%d][0] evaluation loss: %.4f, acc: %.3f" % (epoch, val_loss.avg.item(), 100*val_accuracy.avg.item()))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, _, val_loader = get_datasets(args)
    args.num_steps_per_epoch = len(train_loader)
    model, opt, lr_scheduler, loss_func = get_model(args)

    if args.verbose:
        logger.info("MODEL: %s", args.model)

    start = time.time()

    for epoch in range(args.resume_from_epoch, args.epochs):
        train(epoch, model, opt, lr_scheduler,
             loss_func, train_sampler, train_loader, args)
        if not SPEED:
            validate(epoch, model, loss_func, val_loader, args)
            #save_checkpoint(model, opt, args.checkpoint_format, epoch)

    if args.verbose:
        logger.info("\nTraining time: %s", str(timedelta(seconds=time.time() - start)))
