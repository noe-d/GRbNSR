"""
,---------. .-------.       ____    .-./`) ,---.   .--.    .-''-.  .-------.
\          \|  _ _   \    .'  __ `. \ .-.')|    \  |  |  .'_ _   \ |  _ _   \
 `--.  ,---'| ( ' )  |   /   '  \  \/ `-' \|  ,  \ |  | / ( ` )   '| ( ' )  |
    |   \   |(_ o _) /   |___|  /  | `-'`"`|  |\_ \|  |. (_ o _)  ||(_ o _) /
    :_ _:   | (_,_).' __    _.-`   | .---. |  _( )_\  ||  (_,_)___|| (_,_).' __
    (_I_)   |  |\ \  |  |.'   _    | |   | | (_ o _)  |'  \   .---.|  |\ \  |  |
   (_(=)_)  |  | \ `'   /|  _( )_  | |   | |  (_,_)\  | \  `-'    /|  | \ `'   /
    (_I_)   |  |  \    / \ (_ o _) / |   | |  |    |  |  \       / |  |  \    /    _
    '---'   ''-'   `'-'   '.(_,_).'  '---' '--'    '--'   `'-..-'  ''-'   `'-    _( )_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  loss functions
                                                                                 (_,_)
"""

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .trainer_util import *
import numpy as np

from torch.nn import MSELoss

# Instantiate

def setup_loss_fn(loss_fn,
                  **kwargs
                 ):
    if loss_fn == "mse":
        criterion = MSELoss(**kwargs)
    elif loss_fn == "sce":
        criterion = partial(sce, **kwargs)
    elif criterion == "cluster_consistency":
        criterion = ClusteringConsistency(**kwargs)
    else:
        raise NotImplementedError
    return criterion


# GraphMAE

#adaptation from sce function in GraphMAEModel
class SCELoss(nn.Module):
    def __init__(
        self,
        normalize=True,
        p=2,
        dim=-1,
        alpha=3,
    ):
        super().__init__()
        self._p = p
        self._dim = dim
        self._alpha = alpha

    def forward(self, x, y):
        return partial(sce
                       , alpha=self._alpha
                      )(x, y)


def sce(x, y, p=2, dim=-1, alpha=3):
    x = F.normalize(x, p=p, dim=dim)
    y = F.normalize(y, p=p, dim=dim)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=dim)).pow_(alpha)

    loss = loss.mean()
    return loss


# GCC
"""
class MemoryMoCo(nn.Module):
    #Fixed-size queue with momentum encoder

    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
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
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
"""

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
        else:
            self._device = torch.device("cpu")

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).to(self._device).long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLossNS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
        else:
            self._device = torch.device("cpu")

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        #label = torch.arange(bsz).cuda().long()
        label = torch.arange(bsz).long().to(self._device)
        loss = self.criterion(x, label)
        return loss

    
    
# PGCL

class ClusteringConsistency(nn.Module):
    """ clustering consistency loss """
    
    def __init__(
        self,
        # cluster assignment
        use_queue:bool=True,
        
        epsilon:float=0.05,
        world_size:int=1,
        sinkhorn_iterations:int=3,
        
        # contrast
        temperature:float=0.2,
        hard_selection:bool=True,
        sample_reweighting:bool=False,
    ):
        super(ClusteringConsistency, self).__init__()
        
        self.use_queue = use_queue
        self.epsilon = epsilon
        self.world_size = world_size
        self.sinkhorn_iterations = sinkhorn_iterations
        
        
        self.temperature = temperature
        self.hard_selection = hard_selection
        self.sample_reweighting = sample_reweighting
        
        
    def clustering_consistency_loss(self
                                    , crops_for_assign
                                    , nmb_crops
                                    , queue
                                    , bs
                                    , prototypes
                                    , output
                                    , embedding
                                    ):

        hard_q, neg_q, z = [], [], []
        
        for i, crop_id in enumerate(crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if self.use_queue or not torch.all(queue[i, -1, :] == 0):
                        # print("queue is not None")
                        self.use_queue = True
                        out = torch.cat(
                            (torch.mm(
                                queue[i],
                                prototypes.weight.t()
                            ), 
                             out)
                        )
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                #print(' {} queue : '.format(i), self.queue)
                #print(' out : ', out)
                # get assignments
                q = distributed_sinkhorn(out
                                         , epsilon=self.epsilon
                                         , world_size=self.world_size
                                         , sinkhorn_iterations=self.sinkhorn_iterations
                                        )[-bs:]
                
                #print(' q : ', q)
                # if not i:
                hard_q.append(torch.argmax(q, dim=1))
                ##################### important hyperparameter #####################
                neg_q.append(torch.argsort(q, dim=1)[:, 2:8]) # optional choices
                if not i:
                    pass#global_plabel.append(hard_q[0])
            # cluster assignment prediction
            cc_loss = 0
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                z.append(embedding[bs * v: bs * (v + 1), :] / self.temperature)
                cc_loss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
        
        return cc_loss, z, hard_q, neg_q
    
    def contrast_loss(self
                       , x
                       , x_aug
                       , hard_q
                       , neg_q
                       , prototypes
                      ):
        
        batch_size = x.size(0)

        out_1 = F.normalize(x, dim=1)
        out_2 = F.normalize(x_aug, dim=1)

        # neg score  [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        neg_q = torch.cat([neg_q[0], neg_q[1]], dim=0)
        hard_q = torch.cat([hard_q[0], hard_q[1]], dim=0)
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        sim_matrix  = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

        # compute distances among prototypes
        w = prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        proto_dist = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)

        # proto_dist = F.normalize(proto_dist, dim=1)

        # negative samples selection
        if self.hard_selection:
            for i, row in enumerate(mask):
                for j, col in enumerate(row):
                    if hard_q[j] not in neg_q[i]:
                        mask[i][j] = False

        if self.sample_reweighting:
            reweight = torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
            for i, row in enumerate(sim_matrix):
                for j, col in enumerate(row):
                    if i != j:
                        # obtain the prototype id
                        q_i, q_j = hard_q[i].item(), hard_q[j].item()
                        reweight[i][j] = proto_dist[q_i][q_j]
                reweight[i][i] = torch.min(reweight[i])
                # MaxMin scaler
                r_min, r_max = torch.min(reweight[i]), torch.max(reweight[i])
                reweight[i] = (reweight[i] - r_min) / (r_max - r_min)
            # print("before:{}".format(reweight))

            mu = torch.mean(reweight, dim=1)
            std = torch.std(reweight, dim=1)
            # reweight = (reweight - mu) / std
            reweight = torch.exp(torch.pow((reweight - mu), 2) / (2 * torch.pow(std, 2)))
            # print("after:{}".format(reweight))
            sim_matrix = sim_matrix * reweight

        sim_matrix  = sim_matrix.masked_select(mask)#.view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos = torch.cat([pos, pos], dim=0)

        contrast_loss = (- torch.log(pos / sim_matrix.sum(dim=-1))).mean()
        
        
        return contrast_loss
    
    def forward(self
                , bs
                , crops_for_assign
                , nmb_crops
                , queue
                , prototypes
                , output
                , embedding
               ):
        subloss, z, hard_q, neg_q = self.clustering_consistency_loss(
            bs=bs,
            crops_for_assign=crops_for_assign,
            nmb_crops=nmb_crops,
            queue=queue,
            prototypes=prototypes,
            output=output,
            embedding=embedding,
        )
        
        contrast_loss = self.contrast_loss(
            z[0], z[1], hard_q, neg_q, prototypes
        )
        
        loss = contrast_loss + 6 * subloss / (np.sum(nmb_crops) - 1)
        
        return loss
        
        
        
        