from typing import Any, Dict
import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):

  def __init__(self, params, optimizer_cls, **kwargs):
    """Args:

    params: Iterator of model parameters (all parameters).
    optimizer_cls: The class of the local optimizer (e.g., torch.optim.SGD).
    kwargs: Arguments for the optimizer (lr, weight_decay, etc.).
    """
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    self.global_params = list(params)

    self.local_params = []
    for i, p in enumerate(self.global_params):
      if i % self.world_size == self.rank:
        self.local_params.append(p)

    self.optim = optimizer_cls(self.local_params, **kwargs)
    self.param_groups = self.optim.param_groups
    self.state = self.optim.state
    super().__init__([{"params": []}], defaults={})

  def step(self, closure=None):
    """Performs a single optimization step and synchronizes weights.

    Assumes DDP has already finished (gradients are averaged and present).
    """
    loss = self.optim.step(closure)

    for i, param in enumerate(self.global_params):
      owner = i % self.world_size
      dist.broadcast(param.data, src=owner)

    return loss

  def zero_grad(self, set_to_none=False):
    """Clears gradients.

    Note: We only strictly need to clear 'local_params', but standard practice
    is to clear everything or delegate to the inner optimizer.
    """
    self.optim.zero_grad(set_to_none=set_to_none)

  def add_param_group(self, param_group: Dict[str, Any]):
    """Add a new parameter group to an existing sharded optimizer.

    Args:
        param_group: dictionary containing parameters and optimizer options
    """
    super().add_param_group(param_group)
    # broadcast param_group to all ranks
    new_params = param_group["params"]
    current_global_idx = len(self.global_params)
    self.global_params.extend(new_params)
    local_new_params = []
    for i, p in enumerate(new_params):
      global_idx = current_global_idx + i
      if global_idx % self.world_size == self.rank:
        local_new_params.append(p)
        self.local_params.append(p)

    if local_new_params:
      local_group = param_group.copy()
      local_group["params"] = local_new_params
      self.optim.add_param_group(local_group)

    self.state = self.optim.state
