# Experiments on Cifar-10 and Cifar-100

dnn=resnet32
dataset=cifar10
lr_schedule=cosine
epochs=100
warmup=5
base_lr=0.1

last_batch=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=15 ./train_cifar10.sh &
last_batch=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=16 ./train_cifar10.sh

#last_batch=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=0.1 nworkers=4 lr_schedule=$lr_schedule cutmix=1 autoaugment=1 horovod=0 node_rank=15 ./train_cifar10.sh &
#last_batch=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=0.1 nworkers=4 lr_schedule=$lr_schedule cutmix=1 autoaugment=1 horovod=0 node_rank=16 ./train_cifar10.sh


# finetuning
#dnn=efficientnet-b0
#weight_decay=0.00005
dnn=vit-b16
weight_decay=0

lr_schedule=cosine
epochs=5

#last_batch=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=32 base_lr=0.01 weight_decay=$weight_decay nworkers=4 lr_schedule=$lr_schedule warmup_epochs=0 use_pretrained_model=1 horovod=0 node_rank=15 ./train_cifar10.sh &
#last_batch=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=32 base_lr=0.01 weight_decay=$weight_decay nworkers=4 lr_schedule=$lr_schedule warmup_epochs=0 use_pretrained_model=1 horovod=0 node_rank=16 ./train_cifar10.sh
