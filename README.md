# Last Batch Optimization for Distributed DNN Training

## Introduction
This repository implements last-batch optimization to accelerate distributed DNN training. Specifically, it uses the stale gradient on the last
mini-batch to update the model parameter at the current mini-batch, which enables the last batch gradient aggregation (communication) to be fully overlapped with the current batch gradient computation, including both feed-forward and back-propagation passes. We also propose two simple
but effective training tricks to reduce the impact of using last batch gradient on convergence. 

For more information, please check [PDF](https://github.com/lzhangbv/last-batch-opt/blob/main/LB-OPT.pdf).  

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

## Citation
If you use this code in your work, please cite: 

```
@misc{zhang2022lbopt,
    title = {Last Batch Optimization for Distributed DNN Training},
    author = {Zhang, Lin and Shi, Shaohuai and Li, Bo},
    howpublished = {\url{https://github.com/lzhangbv/last-batch-opt}},
    year=2022,
}
```
