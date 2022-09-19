#!/bin/bash

# model training settings
dnn="${dnn:-resnet50}"
batch_size="${batch_size:-32}"
base_lr="${base_lr:-0.0125}" 
epochs="${epochs:-90}"
warmup_epochs="${warmup_epochs:-5}"
lr_schedule="${lr_schedule:-cosine}"
lr_decay="${lr_decay:-30 60 80}"

# last batch hyper
last_batch="${last_batch:-1}"
sync_warmup="${sync_warmup:-0}"
switch_decay="${switch_decay:-0}"

horovod="${horovod:-0}"
params="--horovod $horovod --model $dnn --batch-size $batch_size --base-lr $base_lr --epochs $epochs --warmup-epochs $warmup_epochs --lr-schedule $lr_schedule --lr-decay $lr_decay --last-batch $last_batch --sync-warmup $sync_warmup --switch-decay $switch_decay --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val"

# multi-node multi-gpu settings
nworkers="${nworkers:-64}"
rdma="${rdma:-1}"

ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

script=examples/pytorch_imagenet_resnet.py

if [ "$horovod" = "1" ]; then
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi

