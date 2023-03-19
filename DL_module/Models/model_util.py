"""
,---.    ,---.    ,-----.     ______         .-''-.    .---.      
|    \  /    |  .'  .-,  '.  |    _ `''.   .'_ _   \   | ,_|      
|  ,  \/  ,  | / ,-.|  \ _ \ | _ | ) _  \ / ( ` )   ',-./  )      
|  |\_   /|  |;  \  '_ /  | :|( ''_'  ) |. (_ o _)  |\  '_ '`)    
|  _( )_/ |  ||  _`,/ \ _/  || . (_) `. ||  (_,_)___| > (_)  )    
| (_ o _) |  |: (  '\_/ \   ;|(_    ._) ''  \   .---.(  .  .-'    
|  (_,_)  |  | \ `"/  \  ) / |  (_.\.' /  \  `-'    / `-'`-'|___  
|  |      |  |  '. \_/``".'  |       .'    \       /   |        \    _ 
'--'      '--'    '-----'    '-----'`       `'-..-'    `--------`  _( )_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Utils (convert, loss, ...)
                                                                   (_,_)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import AvgPooling, Set2Set, MaxPooling, SumPooling
import dgl.function as fn

str2act_dict = {
    'none': nn.Identity,
    'hardtanh': nn.Hardtanh,
    'sigmoid': nn.Sigmoid,
    'relu6': nn.ReLU6,
    'tanh': nn.Tanh,
    'tanhshrink': nn.Tanhshrink,
    'hardshrink': nn.Hardshrink,
    'leakyrelu': nn.LeakyReLU,
    'softshrink': nn.Softshrink,
    'softsign': nn.Softsign,
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'selu': nn.SELU,
}


str2readout_dict = {
    "none":nn.Identity,
    "root":nn.Identity,
    "avg":AvgPooling,
    "sum":SumPooling,
    "max":MaxPooling,
    "set2set":Set2Set
}

def str2act(s, act_args={}):
    
    if s is None:
        return nn.Identity()
    elif s in str2act_dict.keys():
        return str2act_dict[s]()
    elif s in str2act_dict.values():
        return s()
    elif type(s) in [type(a()) for a in str2act_dict.values()]:
        return s
    
    else:
        raise NotImplementedError("Invalid activation function: {}.".format(s))
        
        
def str2readout(s, readout_args={}):
    
    if s is None or s=="root":
        return lambda _, x: x
    elif s in str2readout_dict.keys():
        return str2readout_dict[s](**readout_args)
    elif s in str2readout_dict.values():
        return s(**readout_args)
    elif type(s) in [type(a()) for a in str2act_dict.values()]:
        return s
    
    else:
        raise NotImplementedError("Invalid activation function: {}.".format(s))
        
        
def str2norm(s, norm_args={}):
    
    if s == "layernorm":
        return nn.LayerNorm(**norm_args)
    elif s == "batchnorm":
        return nn.BatchNorm1d(**norm_args)
    elif s == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm", **norm_args)
    else:
        return nn.Identity()
    
    
def str2aggr(s):
    if s == 'sum':
        return fn.sum
    elif s == 'max':
        return fn.max
    elif s == 'mean':
        return fn.mean
    else:
        raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
    
    
    

def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx    
    
    
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def get_x_edges_batch_ng(data, node_num=None):
    if node_num is None:
        node_num = data.num_nodes()

    x = torch.ones((node_num, 1))#.to(device)

    edge_index = data.edges()
    edge_index = torch.stack(list(edge_index), dim=0)

    unbatched_graphs = dgl.unbatch(data)
    batch = torch.Tensor([])
    for i, g in enumerate(unbatched_graphs):
        batch = torch.cat((batch, torch.ones(g.num_nodes(), dtype=int)*i))
    batch = torch.tensor(batch, dtype=int)
    #batch = batch.int()
    num_graphs = len(unbatched_graphs)

    return x, edge_index, batch, num_graphs