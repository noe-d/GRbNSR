"""
,---------. .-------.       ____    .-./`) ,---.   .--.    .-''-.  .-------.
\          \|  _ _   \    .'  __ `. \ .-.')|    \  |  |  .'_ _   \ |  _ _   \
 `--.  ,---'| ( ' )  |   /   '  \  \/ `-' \|  ,  \ |  | / ( ` )   '| ( ' )  |
    |   \   |(_ o _) /   |___|  /  | `-'`"`|  |\_ \|  |. (_ o _)  ||(_ o _) /
    :_ _:   | (_,_).' __    _.-`   | .---. |  _( )_\  ||  (_,_)___|| (_,_).' __
    (_I_)   |  |\ \  |  |.'   _    | |   | | (_ o _)  |'  \   .---.|  |\ \  |  |
   (_(=)_)  |  | \ `'   /|  _( )_  | |   | |  (_,_)\  | \  `-'    /|  | \ `'   /
    (_I_)   |  |  \    / \ (_ o _) / |   | |  |    |  |  \       / |  |  \    /     _
    '---'   ''-'   `'-'   '.(_,_).'  '---' '--'    '--'   `'-..-'  ''-'   `'-     _( )_ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Utils
                                                                                  (_,_)
"""

import pandas as pd
import math

import torch
import torch.nn as nn

# pytorch-template

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# GCC

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)

def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )



class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(
        self
        , inputSize
        , outputSize
        , K
        , T=0.07
        , use_softmax=False
        , todevice='cpu'
    ):
        super().__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self._device = todevice

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv).to(self._device)
        )

        print("Memory MoCo init on device: ", todevice)
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).to(self._device)#.cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

    
# PGCL

@torch.no_grad()
def distributed_sinkhorn(out
                         , epsilon=0.05
                         , world_size=1
                         , sinkhorn_iterations=3
                         , avoid_nan = False # deprecated security
                        ):
    out = out / out.shape[0] # just hack it ?
    #out = out / 100 # just hack it ?
    
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q
    
    if avoid_nan:
        Q = torch.nan_to_num(Q, nan=1.0) # avoid nans ??
        sum_Q = torch.sum(Q)
        Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #sum_of_rows = torch.sum(Q, dim=1)
        #sum_of_rows = torch.reshape(sum_of_rows, (K,1))
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= B
        
        if avoid_nan:
            Q = torch.nan_to_num(Q, nan=0.0) # avoid nans ??
        
        #print(it, sum_of_rows, sum_of_cols)
        #print(torch.sum(Q))

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    
    #print('distributed_sinkhorn - Q : ', Q)
    #print('distributed_sinkhorn - K : ', K)
    #print('distributed_sinkhorn - B : ', B)
    #print('distributed_sinkhorn - sum_of_rows : ', sum_of_rows)
    
    return Q.t()