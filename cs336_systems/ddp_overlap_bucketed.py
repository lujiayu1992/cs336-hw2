import torch
import torch.nn as nn
import torch.distributed as dist

class Bucket:
    def __init__(self, params, bucket_id=0):
        self.params = params            # List[torch.nn.Parameter]
        self.id = bucket_id             # Integer ID for debugging
        
        # State tracking
        self.count = len(params)        # How many params left to compute in this batch?
        self.flat_buffer = None         # The flattened tensor (25MB chunk)
        self.handle = None              # The async network handle (receipt)

    def reset(self):
        """
        Resets the state for the next training iteration.
        MUST be called at the end of finish_gradient_synchronization.
        """
        self.count = len(self.params)
        self.flat_buffer = None
        self.handle = None

    def is_ready(self):
        """Returns True if all parameters in this bucket have finished backward."""
        return self.count == 0
    
class OverlapDDP(nn.Module):
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.model = model
        self.module = model
        self.bucket_size_mb = bucket_size_mb
        self.buckets = []
        self.params = []

        for p in model.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                # Use a closure or partial to capture the specific parameter 'p'
                self.params.append(p)
                # p.register_post_accumulate_grad_hook(self._make_hook(p))
        self.params.reverse()
        self._fill_buckets()

                
    def _make_hook(self, bucket):
        def hook(param):
            bucket.count -= 1
            if not bucket.is_ready():
                return
            
            world_size = dist.get_world_size()
            grads = [p.grad for p in bucket.params]
            bucket.flat_buffer = torch._utils._flatten_dense_tensors(grads)
            bucket.flat_buffer /= world_size
            bucket.handle = dist.all_reduce(bucket.flat_buffer, op=dist.ReduceOp.SUM, async_op=True)
        return hook
    
    def _fill_buckets(self):
        current_bucket_size = 0
        current_bucket_params = []
        for p in self.params:
            if current_bucket_size< self.bucket_size_mb:
                current_bucket_size += p.numel()* p.element_size()/ (1024 * 1024)
                current_bucket_params.append(p)
            else:
                self.buckets.append(Bucket(current_bucket_params, len(self.buckets)))
                current_bucket_size = p.numel()* p.element_size()/ (1024 * 1024)
                current_bucket_params = [p]
        if current_bucket_params:
            self.buckets.append(Bucket(current_bucket_params, len(self.buckets)))
        for bucket in self.buckets:
            for p in bucket.params:
                p.register_post_accumulate_grad_hook(self._make_hook(bucket))


    def forward(self, *inputs, **kwargs):
        """
        Standard forward pass. 
        Since this is a wrapper, we just pass inputs to the underlying model.
        """
        return self.model(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Waits for all asynchronous communication to finish.
        MUST be called after loss.backward() and before optimizer.step().
        """
        for bucket in self.buckets:
            if bucket.is_ready():
                bucket.handle.wait()
                grads = [p.grad for p in bucket.params]
                restored_grads = torch._utils._unflatten_dense_tensors(bucket.flat_buffer, grads)
                for p, synced_grad in zip(bucket.params, restored_grads):
                    p.grad.data.copy_(synced_grad)
                bucket.reset()
            