#!/bin/bash

# model training settings
dataset=cifar10
dnn="${dnn:-resnet110}"

# first-order hyper
batch_size="${batch_size:-128}"
base_lr="${base_lr:-0.1}"
lr_schedule="${lr_schedule:-step}"
lr_decay="${lr_decay:-0.5 0.75}"

epochs="${epochs:-100}"
warmup_epochs="${warmup_epochs:-5}"

momentum="${momentum:-0.9}"
use_adam="${use_adam:-0}"
weight_decay="${weight_decay:-0.0005}"

# tricks
label_smoothing="${label_smoothing:-0}"
mixup="${mixup:-0}"
cutmix="${cutmix:-0}"
cutout="${cutout:-0}"
autoaugment="${autoaugment:-0}"
use_pretrained_model="${use_pretrained_model:-0}"

# last batch hyper
last_batch="${last_batch:-1}"

horovod="${horovod:-0}"
params="--dataset $dataset --dir /datasets/cifar10 --model $dnn --batch-size $batch_size --base-lr $base_lr --lr-schedule $lr_schedule --lr-decay $lr_decay --epochs $epochs --warmup-epochs $warmup_epochs --momentum $momentum --use-adam $use_adam --weight-decay $weight_decay --label-smoothing $label_smoothing --mixup $mixup --cutmix $cutmix --autoaugment $autoaugment --cutout $cutout --use-pretrained-model $use_pretrained_model --last-batch $last_batch"

nworkers="${nworkers:-4}"
rdma="${rdma:-1}"
clusterprefix="${clusterprefix:-cluster}"

ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

script=examples/pytorch_cifar10_resnet.py

if [ "$horovod" = "1" ]; then
clusterprefix=$clusterprefix nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
