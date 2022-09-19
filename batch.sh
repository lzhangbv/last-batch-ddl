# Experiments on Cifar-10 and Cifar-100

# S-SGD vs. LB-SGD
epochs=100
base_lr=0.1
lr_schedule=cosine
warmup=5

dnn=resnet32
dataset=cifar10
#last_batch=0 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=1 ./train_cifar10.sh & 
#last_batch=1 sync_warmup=1 switch_decay=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=2 ./train_cifar10.sh &
#last_batch=1 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=3 ./train_cifar10.sh &
#last_batch=1 sync_warmup=1 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=4 ./train_cifar10.sh &

dnn=vgg16
dataset=cifar10
#last_batch=0 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=5 ./train_cifar10.sh & 
#last_batch=1 sync_warmup=1 switch_decay=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=6 ./train_cifar10.sh &
#last_batch=1 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=7 ./train_cifar10.sh &
#last_batch=1 sync_warmup=1 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=8 ./train_cifar10.sh &

dnn=resnet32
dataset=cifar100
#last_batch=0 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=9 ./train_cifar10.sh & 
#last_batch=1 sync_warmup=1 switch_decay=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=10 ./train_cifar10.sh &
#last_batch=1 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=11 ./train_cifar10.sh &
#last_batch=1 sync_warmup=1 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=12 ./train_cifar10.sh &

dnn=vgg16
dataset=cifar100
#last_batch=0 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=13 ./train_cifar10.sh & 
#last_batch=1 sync_warmup=1 switch_decay=1 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=14 ./train_cifar10.sh &
#last_batch=1 sync_warmup=0 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=15 ./train_cifar10.sh &
#last_batch=1 sync_warmup=1 switch_decay=0 dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=16 ./train_cifar10.sh &

# other optimizers: Adagrad, NAG, Adam, AdamW
dnn=resnet32
dataset=cifar10
#dnn=vgg16
#dataset=cifar100

opt_name=adagrad
base_lr=0.01
#last_batch=0 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=1 ./train_cifar10.sh & 
#last_batch=1 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=2 ./train_cifar10.sh &

opt_name=nag
base_lr=0.1
#last_batch=0 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=3 ./train_cifar10.sh & 
#last_batch=1 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=4 ./train_cifar10.sh &

opt_name=adam
base_lr=0.001
#last_batch=0 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=5 ./train_cifar10.sh & 
#last_batch=1 opt_name=$opt_name dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=6 ./train_cifar10.sh &

opt_name=adamw
base_lr=0.001
weight_decay=0.05
#last_batch=0 opt_name=$opt_name weight_decay=$weight_decay dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=7 ./train_cifar10.sh & 
#last_batch=1 opt_name=$opt_name weight_decay=$weight_decay dnn=$dnn dataset=$dataset epochs=$epochs batch_size=128 base_lr=$base_lr nworkers=4 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=8 ./train_cifar10.sh &


# Experiemnts on ImageNet
epochs=90
base_lr=0.1
lr_schedule=cosine
warmup=5

dnn=resnet50
#last_batch=0 sync_warmup=0 switch_decay=0 dnn=$dnn epochs=$epochs batch_size=32 base_lr=$base_lr nworkers=32 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=1 node_count=8 ./train_imagenet.sh
#last_batch=1 sync_warmup=1 switch_decay=1 dnn=$dnn epochs=$epochs batch_size=32 base_lr=$base_lr nworkers=32 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=1 node_count=8 ./train_imagenet.sh
last_batch=1 sync_warmup=0 switch_decay=0 dnn=$dnn epochs=$epochs batch_size=32 base_lr=$base_lr nworkers=32 lr_schedule=$lr_schedule warmup_epochs=$warmup horovod=0 node_rank=1 node_count=8 ./train_imagenet.sh

