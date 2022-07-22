# Copyright 2020 HKBU. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Fully Pipelined Data Parallelism built on pure PyTorch (WIP). 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import collections
import numpy as np
import torch.distributed as dist

def rank():
    return dist.get_rank()

def size():
    return dist.get_world_size()

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression):
        """
        Distributed optimizer with fully pipelined data parallelism.
        """
        super(self.__class__, self).__init__(params)
        self._num_steps = 0

        # parameter names
        if named_parameters is not None:
            self._param_names = {v: k for k, v in sorted(named_parameters)}
        else:
            self._param_names = {v: 'param.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        
        if dist.get_world_size() > 1:
            # Buffer for grad reduction.
            self._prepare_grad_buffer()
            # Stream for grad reduction.  
            self._comm_stream = torch.cuda.Stream()

    def _prepare_grad_buffer(self):
        """
        Prepare buffer to store all grads for grad reduction.
        """
        self._grad_buffer = None
        self._param_buffer_idx = {}
        self._register_parameters = []

        start_p = 0
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    param_name = self._param_names[p]
                    numel = p.data.numel()
                    self._param_buffer_idx[param_name] = (start_p, start_p+numel) # get param idx by name
                    self._register_parameters.append(p)
                    start_p += numel
        
        with torch.no_grad():
            self._grad_buffer = p.data.new_zeros(start_p)

        if dist.get_rank() == 0: 
            print('Buffer size (MB): %.2f' % (self._grad_buffer.numel()*4/1024/1024))
   
    def _update_grad_with_buffer(self):
        """Update p.grad with buffer, and update buffer with p.grad."""
        for p in self._register_parameters:
            # get reduced grad from buffer
            param_name = self._param_names[p]
            start_p, end_p = self._param_buffer_idx[param_name]
            reduced_grad = self._grad_buffer[start_p:end_p]
            reduced_grad.div_(dist.get_world_size())
            # swap: reduced_grad -> p.grad for param update; p.grad -> buffer for grad reduction. 
            self._swap_tensors(reduced_grad, p.grad.data)
            
    def _swap_tensors(self, reduced_grad, computed_grad):
        tmp = computed_grad.clone().detach()
        computed_grad.view(-1).copy_(reduced_grad)
        reduced_grad.copy_(tmp.view(-1))
        del tmp

    def step(self, closure=None):
        """Performs a single optimization step."""
        
        if dist.get_world_size() > 1:
            # sync grad reduction in the buffer
            if self._num_steps > 0:
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            
            # swap p.grad and reduced grad in the buffer
            self._update_grad_with_buffer()

            # start all-reducing new grads in the buffer
            self._comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._comm_stream):
                dist.all_reduce(self._grad_buffer)

        super(self.__class__, self).step(closure)
        self._num_steps += 1

    def synchronize(self):
        if dist.get_world_size() > 1:
            torch.cuda.synchronize()


def DistributedOptimizer(optimizer, named_parameters, compression=None):
    """
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    
    Usage Example:
        for data, target in train_loader:
            optim.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optim.step()

    Gradient Accumulation Example:
        for data, target in train_loader:
            optim.zero_grad()
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i+batch_size]
                target_batch = target[i:i+batch_size]
                output = model(data_batch)
                loss = criterion(output, target_batch)
                loss.div_(math.ceil(float(len(data)) / batch_size))
                loss.backward()
            optim.step()
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression)

def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    for name, p in params:
        if p is not None:
            dist.broadcast(p.view(-1), root_rank)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if p is not None and not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, _ in params:
        if key in callbacks:
            callbacks[key]()

def allreduce(tensor, name=None):
    dist.all_reduce(tensor)
    return tensor.div_(dist.get_world_size())
