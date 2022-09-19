# Last Batch Optimization for Distributed DNN Training

## Introduction
This repository implements last-batch optimization to accelerate distributed DNN training 
by fully pipelining the feed-forward and back-propagation computations with the last-batch gradient communications. 

## Usage

The last batch optimizer can be easily added to exisiting training scripts.

```Python
from opt import fpdp as hvd
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = hvd.DistributedOptimizer(optimizer, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
...
```

Note that `fpdp` stands for fully pipelined data parallelism, and the usage of our optimizer is similar to `horovod.DistributedOptimizer()`. 

## Configure the cluster settings

Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster enviroments

## Run experiments

```
$ mkdir logs
$ bash batch.sh
```
