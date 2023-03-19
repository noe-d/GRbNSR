"""
 ______        ____   ,---------.    ____
|    _ `''.  .'  __ `.\          \ .'  __ `.
| _ | ) _  \/   '  \  \`--.  ,---'/   '  \  \
|( ''_'  ) ||___|  /  |   |   \   |___|  /  |
| . (_) `. |   _.-`   |   :_ _:      _.-`   |
|(_    ._) '.'   _    |   (_I_)   .'   _    |
|  (_.\.' / |  _( )_  |  (_(=)_)  |  _( )_  |    
|       .'  \ (_ o _) /   (_I_)   \ (_ o _) /   _
'-----'`     '.(_,_).'    '---'    '.(_,_).'  _( )_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _) utils to load and process data
                                              (_,_)
"""
from .data_loader import *

from tqdm import tqdm
import numpy as np
import math
import pandas as pd
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing

import graph_tool.all as gt
from graph_tool.spectral import adjacency

import torch
import torch.nn.functional as F

import dgl
from dgl import RemoveSelfLoop
from dgl.data import TUDataset

import logging

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_TU_NAMES = [ # non-exhaustive list
    "REDDIT-BINARY"
    , "IMDB-BINARY"
    , "IMDB-MULTI"
    , "DD"
    , "ENZYMES"
    , "COLLAB"
    , "PROTEINS"
    , "REDDIT-MULTI-5K"
    , "MUTAG"
]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#   _( )_  ===========================================================================
#  (_ o _)                       make / convert classes
#   (_,_)  ===========================================================================

def load_data(dataset_str,
              name_col = None,
              label_col=None,
              preprocess=True,
             ):
    split_str = dataset_str.split(" ")
    if len(split_str)>1:
        dataset_name, dataset_len = split_str[0], int(split_str[1])
    else:
        dataset_name, dataset_len = split_str[0], -1

    rm_graphs = []
    if dataset_name in _TU_NAMES:
        tu_data = TUDataset(dataset_name)
        
        dgl_graphs = list(tu_data.graph_lists)
        graphs_labels = [int(lab[0]) for lab in list(tu_data.graph_labels)]
        
        if dataset_len>0:
            dgl_graphs=list(dgl_graphs)[:dataset_len]
            graphs_labels=list(graphs_labels)[:dataset_len]
            
        
        src_format = "DGL"
        from_csv = False
    
    elif dataset_name.endswith(".csv"): # consider it from graph-tool
        src_format = "graph-tool"
        from_csv = True
        
        gt_graphs_names, graphs_labels = load_from_csv(dataset_name
                                                       , name_col=name_col
                                                       , label_col=label_col
                                                      )
                         
        if dataset_len>0:
            gt_graphs_names=list(gt_graphs_names)[:dataset_len]
            graphs_labels=list(graphs_labels)[:dataset_len]
        
        dgl_graphs, rm_graphs = graphtoolkeys2dglgraphs(gt_graphs_names)
        # remove graph that could not be loaded
        if len(rm_graphs)>0:
            cleand_g_labels = []
            cleaned_g_names = []
            for lab, n in zip(graphs_labels, gt_graphs_names):
                if not n in rm_graphs:
                    cleand_g_labels += [lab]
                    cleaned_g_names += [n]

            graphs_labels = cleand_g_labels
            gt_graphs_names = cleaned_g_names
        
    else:
        raise NotImplementedError("Error testing {} dataset".format(dataset_name))
        
    # always add reverse-edges to convert directed graphs to BIdirected graphs
    # that can be treated as undirected graphs + remove self-loops and duplicates
    # (DGL was developped to handle directed graphs not undirected ones)
    if preprocess:
        rmv_loops = RemoveSelfLoop()
        dgl_graphs = [dgl.to_simple(rmv_loops(dgl.add_reverse_edges(g))) for g in dgl_graphs]
    
    return dataset_name, from_csv, src_format, dgl_graphs, graphs_labels, rm_graphs

def load_from_csv(csv_info:str
                  , name_col:str="Name"
                  , label_col:str=None
                 ):
    csv_info = csv_info.split(" ")
    csv_path = csv_info[0]
    n_loaded = -1
    if len(csv_info)>1:
        n_loaded = int(csv_info[1])
    data_info = pd.read_csv(csv_path)
    
    dgl_graphs = list(data_info[name_col])[:n_loaded]
    
    graphs_labels = None
    if label_col is not None:
        graphs_labels = list(data_info[label_col])[:n_loaded]
    
    return dgl_graphs, graphs_labels


def graphtoolkeys2dglgraphs(gt_keys:list,
                            verbosity:bool=True
                           ):
    """
    TODOs:
        - [x] handle non healthy graphs (that can't be loaded)
        - [ ] 
    """
    try:  
        logger = logging.getLogger('train')
        display_logger = logger.info
    except:
        display_logger = print
        
    
    if verbosity: 
        display_logger("🦦 converting {n_g} graphs from graph-tool library to DGL format.".format(n_g=len(gt_keys)))
          
    # load graphs from graph-tool library and convert them to dgl.DGLGraph 
    dgl_list = []
    gt_unhealthy = []
    
    pbar = tqdm(gt_keys)
    for k in pbar:
        pbar.set_postfix({'graph': k})
        try:
            dgl_list += [dgl.from_scipy(adjacency(gt.collection.ns[k]))]
        except:
            gt_unhealthy += [k]
            
    if len(gt_unhealthy)>0:
        display_logger("The following graphs:\n'{}'\n... could not have been loaded from graph-tool library.".format(gt_unhealthy))
    else:
        display_logger("Whole dataset of {} graphs loaded successfully.".format(len(dgl_list)))
    
    return dgl_list, gt_unhealthy
    
# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GCC UTILS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# see: https://github.com/THUDM/GCC/blob/20398aac95957784865d6c78bc46ead605221f0d/gcc/
#     datasets/data_util.py
# ====================================================================================

def _rwr_trace_to_dgl_graph(g
                            , seed
                            , trace
                            , positional_embedding_size
                            , entire_graph=False
                           ):
    subv = torch.unique(torch.cat(trace)).tolist()
    subv = [s for s in subv if s>0]
    #print("Nb sub vertices : ", len(subv))
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)
        
    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float) --> DEPRECATED
    adj = g.adjacency_matrix(transpose=False, scipy_fmt="csr").astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = sparse.linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
            
    # drafty fix: replacing NaNs by 0s ...
    u = np.nan_to_num(u)
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


# ====================================================================================
#                                   GraphMAE UTILS 
# see: customized from graphmae/datasets/data_util
#       - [ ] make degrees feat to unload class definition
#       - [ ] 
# ====================================================================================


# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PGCL UTILS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# augment graphs --> adapt to fit the DGL framework 
#       - [x] subgraph
#       - [x] stro_drop_nodes
#       - [...] others
#             - [x] drop_nodes, drop_edge_nodes, permute_edges, stro_subgraph
#             - [ ] mask_nodes, ppr_aug, rotate, clip
# ====================================================================================

def subgraph(graph):

    #node_num, _ = data.x.size()
    node_num = graph.num_nodes()
    #_, edge_num = data.edge_index.size()
    edge_num = graph.num_edges()
    
    sub_num = int(node_num * 0.2)

    #edge_index = data.edge_index.numpy()
    edge_index = graph.edges()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    #edge_index = data.edge_index.numpy()
    edge_index = graph.edges()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    #data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           ,torch.Tensor(edge_index[1])),
                          num_nodes = node_num
                         )
    
    return out_graph

def stro_subgraph(data):
    node_num = graph.num_nodes()
    edge_num = graph.num_edges()
    
    sub_num = int(node_num * 0.1)
    edge_index = graph.edges()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        # if count > node_num:
        #     break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        # print('[info] stro idx_neigh is:{}'.format(idx_neigh))
        # print('[info] stro sample_node is:{}'.format(sample_node))

        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = graph.edges()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_nondrop, :] = 0
    adj[:, idx_nondrop] = 0
    edge_index = adj.nonzero().t()

    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           ,torch.Tensor(edge_index[1])),
                          num_nodes = node_num
                         )
    
    return out_graph


def drop_nodes(graph):
    
    #node_num, _ = data.x.size()
    node_num = graph.num_nodes()
    #_, edge_num = data.edge_index.size()
    edge_num = graph.num_edges()

    drop_num = int(node_num / 10)     # ratio for remained nodes
    # print('[info] drop_num is:{}'.format(drop_num))

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    # print('[info] idx_drop is:{}'.format(idx_drop))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = graph.edges()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           , torch.Tensor(edge_index[1]))
                          , num_nodes = node_num
                         )
    
    return out_graph

def drop_edge_nodes(graph):
    
    #node_num, _ = data.x.size()
    node_num = graph.num_nodes()
    #_, edge_num = data.edge_index.size()
    edge_num = graph.num_edges()
    
    ratio = 0.1
    
    degrees = dict(zip(graph.nodes(), graph.in_degrees()))
    degrees = sorted(degrees.items(), key = lambda item: item[1])

    idx_drop = np.array([n[0] for n in degrees[:int(len(degrees) * ratio)]])
    drop_num = len(idx_drop)
    drop_ratio = drop_num / node_num
    # print('drop_ratio is: {}'.format(drop_ratio))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    edge_idx_drop = [idx for idx, n in enumerate(graph.edges()[0]) if n.item() in idx_drop or graph.edges()[1][idx].item() in idx_drop]
    edge_idx_nondrop = [n for n in range(len(graph.edges()[0])) if not n in edge_idx_drop]

    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = graph.edges()#data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    #data.edge_index = edge_index
    #try:
    #    data.edge_attr = data.edge_attr[edge_idx_nondrop]
    #except:
    #    pass
    
    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           , torch.Tensor(edge_index[1]))
                          , num_nodes = node_num
                         )
    
    return out_graph


def permute_edges(graph):

    node_num = graph.num_nodes()
    edge_num = graph.num_edges()
    permute_num = int(edge_num / 10)

    edge_index = graph.edges()
    edge_index = torch.stack(list(edge_index), dim=0)
    edge_index = edge_index.transpose(0, 1)

    idx_add = np.random.choice(node_num, (permute_num, 2))

    rand_ids = np.random.choice(edge_num, edge_num-permute_num, replace=False)
    edge_index = edge_index[rand_ids]

    #data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    #try:
    #    data.edge_attr = data.edge_attr[edge_idx_nondrop]
    #except:
    #    pass
    
    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           , torch.Tensor(edge_index[1]))
                          , num_nodes = node_num
                         )
    
    return out_graph

def stro_drop_nodes(graph):
    
    #node_num, _ = data.x.size()
    node_num = graph.num_nodes()
    #_, edge_num = data.edge_index.size()
    edge_num = graph.num_edges()
    
    degrees = dict(zip(graph.nodes(), graph.in_degrees()))
    for key, val in degrees.items():
        degrees[key] = math.log(val+1)
        
    min_d, max_d = min(degrees.values()), max(degrees.values())
    if min_d==max_d:
    #idx_drop = np.array([n for n in degrees if (degrees[n] - min_d) / (max_d - min_d) <= 0.2])
        idx_drop = np.array([n for n in degrees if np.random.random() <= 0.2])
    else:
        idx_drop = np.array([n for n in degrees if (degrees[n] - min_d) / (max_d - min_d) <= 0.2])
    
    drop_num = len(idx_drop)
    drop_ratio = len(idx_drop) / node_num
    
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = graph.edges()
    
    
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    out_graph = dgl.graph((torch.Tensor(edge_index[0])
                           , torch.Tensor(edge_index[1]))
                          , num_nodes = node_num
                         )
    
    return out_graph



# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ====================================================================================

def make_table(stats_dict:dict
               , name:str="Recap Table"
               , colwidth:int = 60
               , key_width:int = 40
               , sep:str="|"
              ):
    if len(sep)>1:
        sep = "|"
    if key_width>colwidth:
        key_width = colwidth//3
    if not (len(name)+colwidth)%2==0:
        colwidth -=1
    val_width = colwidth-key_width
    
    
    toprule = "┌{}┐\n".format("-"*(colwidth))
    midrule = "├{}┤\n".format("-"*(colwidth))  
    bottomrule = "└{}┘\n".format("-"*(colwidth)) 
    
    header = "|{ws}{n}{ws}|\n".format(n=name, ws=" "*((colwidth-len(name))//2))
    
    tablecore = ""
    for k, v in stats_dict.items():
        row = "|{key}{wsk}{s} {wsv}{val}|\n".format(key=k, val=v
                                                    , wsk = " "*(key_width-len(k)-1)
                                                    , wsv = " "*(val_width-len(str(v))-1)
                                                    , s=sep
                                                   )
        tablecore += row
        
    
    table = toprule+header+midrule+tablecore+bottomrule
    
    return table


# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GraphDataLoader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ====================================================================================

# collate functions
# =================

def collate_fn_graphmae(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [torch.Tensor([x[1]]) for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels

def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

    return batcher_dev

def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev

str2collate = {
    "collate_fn_graphmae":collate_fn_graphmae,
    "collate_fn_gcc":batcher(),
    "batcher":batcher(),
    "labeled_batcher":labeled_batcher(),
}

def get_collate_fn(collate_str):
    if not collate_str in str2collate.keys():
        raise NotImplementedError("{ci} collate function is not implemented. Implemented methods are: {cl}.".format(ci=collate_str, cl=str2collate.keys()))
        
    else:
        return str2collate[collate_str]
        
# sampler functions
# =================

from torch.utils.data.sampler import SubsetRandomSampler

str2sampler = {
    "SubsetRandomSampler":SubsetRandomSampler
}