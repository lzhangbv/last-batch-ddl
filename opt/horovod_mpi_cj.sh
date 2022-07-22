#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
exclude_parts="${exclude_parts:-''}"
accum="${accum:-1}"
source ../configs/envs.conf

if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    script=bert_benchmark.py
    params="--model $dnn --sentence-len $senlen --batch-size $bs --batches-per-allreduce $accum"
else
    script=imagenet_benchmark.py
    params="--model $dnn --batch-size $bs --batches-per-allreduce $accum"
fi


# multi-node multi-GPU setting
node_rank=1  # launch node1, node2, ...
ngpu_per_node=4
node_count=$(expr $nworkers / $ngpu_per_node)

if [ $nworkers -lt 4 ]; then # single-node
    ngpu_per_node=$nworkers node_count=1 node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
else
    ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
fi

