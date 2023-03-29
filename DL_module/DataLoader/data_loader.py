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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _) Data classes
                                              (_,_)
"""

from . import data_util

from tqdm import tqdm
import numpy as np
import math
import pandas as pd

import dgl
from dgl.dataloading import GraphDataLoader # sub-class of torch.utils.data.DataLoader

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# ====================================================================================
# XXXXXXXX  Graph — DEPRECATED
# ====================================================================================

class Graph(dgl.DGLGraph):
    def __init__(
        self,
        name:str=None,
        label:str=None
    ):
        self.name = name
        self.label = label
        


# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ====================================================================================

class GraphDataset(Dataset):
    def __init__(
        self,
        dgl_graphs:list,
        graphs_labels:list=None,
        og_format:str="dgl",
        verbosity:bool=False,
        dataset_name:str="GraphDataset",
        display_col_width:int=60,
        
        from_csv:bool=False, # ??
        name_col:str="Name",
        label_col:str="Category",
    ):
        if isinstance(dgl_graphs[0], str):
            dataset_name, from_csv, og_format, dgl_graphs, graphs_labels, rm_graphs = data_util.load_data(
                dgl_graphs,
                name_col=name_col,
                label_col=label_col,
            )
            self.non_loaded_graphs = rm_graphs
        
        self.name = dataset_name
        self._og_format = og_format
        self.graphs = dgl_graphs
        if graphs_labels is not None:
            assert len(graphs_labels) == len(self.graphs), "Number of labels ({n_lab}) doesn't match the number of graphs ({n_g}).".format(n_lab=len(graphs_labels), n_g=len(self.graphs))
            self.num_labels = len(set(graphs_labels))
            
            # encode labels
            unique_labels = list(set(graphs_labels))
            unique_labels.sort() # keep order if numerical labels
            self.labId_to_labName = dict(enumerate(unique_labels)) # {id: name}
            self.labName_to_labId = {v: k for k, v in self.labId_to_labName.items()}
            
            graphs_labels = [self.labName_to_labId[lab] for lab in graphs_labels]
            
        self.graphs_labels = graphs_labels
        
        self._has_labels = not (self.graphs_labels is None)
            
        self._display_col_width=display_col_width
        #if verbosity: print(self.display_statistics())
        
    def __len__(
        self,
    ):
        return len(self.graphs)
    
    def __getitem__(
        self,
        idx
    ):
        return self.graphs[idx]
    
    
    def __str__(self):
        return self.display_statistics()
    
    def display_statistics(self,):
        nodes = [graph.num_nodes() for graph in self.graphs]
        edges = [graph.num_edges() for graph in self.graphs]
        mean_degrees = [np.mean(np.array(graph.in_degrees())) for graph in self.graphs]
        
        stats = {
            "number of graphs": len(self.graphs),
            
            "nodes — tot": np.sum(nodes),
            "nodes — mean": np.mean(nodes),
            "nodes — median": np.median(nodes),
            "nodes — min": np.min(nodes),
            "nodes — max": np.max(nodes),
            
            "edges — tot": np.sum(edges),
            "edges — mean": np.mean(edges),
            "edges — median": np.median(edges),
            "edges — min": np.min(edges),
            "edges — max": np.max(edges),
            
            #"av. deg — tot": np.sum(mean_degrees),
            "av. deg — mean": np.mean(mean_degrees),
            "av. deg — median": np.median(mean_degrees),
            "av. deg — min": np.min(mean_degrees),
            "av. deg — max": np.max(mean_degrees),
        }
        
        if self.graphs_labels is not None:
            stats["number of labels"] = self.num_labels
            for lab in set(self.graphs_labels):
                n_lab = np.sum([lab==l for l in self.graphs_labels])
                prop_lab = n_lab/len(self)
                lab_str = "   - {labN} ({labI})".format(labN=lab, labI=self.labId_to_labName[lab])
                stats[lab_str] = "{n_gs} ({prop:.3g} %)".format(n_gs=n_lab, prop=prop_lab*100)
        
        
        return data_util.make_table(stats
                                     , name=self.name
                                     , colwidth=self._display_col_width
                                    )
    
    # --------------------------------
    #    build the GraphDataLoader   
    # --------------------------------
    
    def make_GraphDataLoader(
        self,
        loader_config:dict={},
        validation:bool=False,
    ):
        graphdataloader_configs = loader_config.copy()
        
        if validation:
            graphdataloader_configs.pop("sampler")
            graphdataloader_configs["shuffle"]=False
            if "val_collate_fn" in graphdataloader_configs.keys():
                graphdataloader_configs["collate_fn"] = graphdataloader_configs["val_collate_fn"]
        
        if "collate_fn" in graphdataloader_configs.keys():
            graphdataloader_configs["collate_fn"] = data_util.get_collate_fn(graphdataloader_configs["collate_fn"])
            
        if "sampler" in graphdataloader_configs.keys():
            if not validation:
                graphdataloader_configs["sampler"] = self.get_sampler_fn(graphdataloader_configs["sampler"])
        
        return GraphDataLoader(
            dataset=self,
            **graphdataloader_configs
        )
    
    
    def get_sampler_fn(self, sampler_str):
        if sampler_str=="SubsetRandomSampler":
            idx = torch.arange(self.__len__())
            sampler = data_util.SubsetRandomSampler(idx)
            return sampler

        else:
            raise NotImplementedError("{si} sampler function is not implemented.".format(si=sampler_str))
    
# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GCC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# see: https://github.com/THUDM/GCC/blob/20398aac95957784865d6c78bc46ead605221f0d/gcc/
#      datasets/graph_dataset.py#L312
# ====================================================================================
    
class GCCDataset(GraphDataset):
    def __init__(
        self,

        dgl_graphs:list,
        graphs_labels:list=None,
        verbosity:bool=False,
        dataset_name:str="GCCDataset",
        
        from_csv:bool=False,
        name_col:str="Name",
        label_col:str="Category",
        
        rw_hops=64,
        subgraph_size=None,
        restart_prob=0.001,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        use_entire_graph=True,
    ):
        super().__init__(
            dgl_graphs=dgl_graphs,
            graphs_labels=graphs_labels,
            verbosity=verbosity,
            dataset_name=dataset_name,
            from_csv=from_csv,
            name_col=name_col,
            label_col=label_col,
        )
        self.rw_hops=rw_hops
        self.subgraph_size=subgraph_size
        self.restart_prob=restart_prob
        self.positional_embedding_size=positional_embedding_size
        self.step_dist=step_dist
        
        self.entire_graph = use_entire_graph
        
        assert sum(step_dist) == 1.0, "< step_dist > must sum to 1.0"
        assert positional_embedding_size > 1, "< positional_embedding_size > must be > 1."
        
        
    def _convert_idx(self, idx):
        """
        from GraphClassificationDataset
        """
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()
        return graph_idx, node_idx
    
    
    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.sampling.random_walk(
                g=self.graphs[graph_idx], 
                nodes=[node_idx], 
                num_traces=1,
                length=step # ??
                #num_hops=step
            )[0][0][-1].item()

        # added custom subg size
        max_nodes_per_seed = None
        if hasattr(self, "subgraph_size"):
            max_nodes_per_seed = self.subgraph_size
        if max_nodes_per_seed is None and self.restart_prob>0:
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                        self.graphs[graph_idx].out_degrees(node_idx) #DEPRECATED: out_degree --> out_degrees
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
        else:
            max_nodes_per_seed = self.rw_hops
        traces = dgl.sampling.random_walk(
            g = self.graphs[graph_idx],
            nodes=[node_idx, other_node_idx],
            #restart_prob=self.restart_prob,
            length=max_nodes_per_seed, # MODIFIED
        )
        #print("Nb nodes : {n} | degs: {d} | max: {m}".format(n=self.graphs[graph_idx].num_nodes(),d=self.graphs[graph_idx].out_degrees()[node_idx], m=max_nodes_per_seed))

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=(traces[0][0],),  # MODIFIED
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=(traces[0][1],),  # MODIFIED
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k
    
    
    def _create_dgl_graph(self, data):
        """
        from NodeClassificationDataset
        """
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return
        
        
# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GraphMAE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# see: https://github.com/THUDM/GraphMAE/blob/
#      fdea388026e00d84a706e06e3c1e772aae0e6fd7/graphmae/datasets/data_util.py
# ====================================================================================
    
class GraphMAEDataset(GraphDataset):
    """
    in og code:
        is an instance of dgl.data.TUDataset --> DGLBuiltinDataset
    """
    def __init__(
        self,

        dgl_graphs:list,
        graphs_labels:list=None,
        verbosity:bool=False,
        dataset_name:str="GraphMAEDataset",
        
        from_csv:bool=False,
        name_col:str="Name",
        label_col:str="Category",
        
        deg4feat:bool=True,
        MAX_DEGREES:int=400,
        feature_dim:int=None,
    ):
                
        super().__init__(
            dgl_graphs=dgl_graphs,
            graphs_labels=graphs_labels,
            verbosity=verbosity,
            dataset_name=dataset_name,
            from_csv=from_csv,
            name_col=name_col,
            label_col=label_col,
        )
        
        self.degrees = [g.in_degrees() for g in self.graphs]
        max_degs = [np.max(degs.tolist()) for degs in self.degrees]
        
        if feature_dim is None:
            overall_max_deg = np.max(max_degs)
            self.feature_dim = int(np.min([overall_max_deg, MAX_DEGREES]) + 1)
        else: 
            self.feature_dim = feature_dim
            
        self.oversize = np.sum([max_deg>MAX_DEGREES for max_deg in max_degs])
        
        if deg4feat:
            for i, degs in enumerate(self.degrees):
                deg_roofed = degs
                deg_roofed[deg_roofed>self.feature_dim-1] = self.feature_dim-1
                degs = deg_roofed

                feat = F.one_hot(degs, num_classes=self.feature_dim).float()
                self.graphs[i].ndata["attr"] = feat
        
        
    def __getitem__(self, idx):
        if self._has_labels:
            return self.graphs[idx], self.graphs_labels[idx]
        else:
            return self.graphs[idx]
        
        
# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PGCL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# see: https://github.com/ha-lins/PGCL/blob/master/unsupervised_TU/aug.py
# ====================================================================================

class PGCLDataset(GraphDataset):
    def __init__(
        self,

        # SUPER 
        
        dgl_graphs:list,
        graphs_labels:list=None,
        verbosity:bool=False,
        dataset_name:str="PGCLDataset",
        
        from_csv:bool=False,
        name_col:str="Name",
        label_col:str="Category",
        
        # PGCL
        aug:str='subgraph',
        stro_aug:str='stro_dnodes',
        weak_aug2:str=None,
        
    ):
        super().__init__(
            dgl_graphs=dgl_graphs,
            graphs_labels=graphs_labels,
            verbosity=verbosity,
            dataset_name=dataset_name,
            from_csv=from_csv,
            name_col=name_col,
            label_col=label_col,
        )
        
        self.aug = aug
        self.stro_aug = stro_aug
        self.weak_aug2 = weak_aug2
        
    
    def __getitem__(self, idx):
        return self.get(idx)
    
    def get(self, idx):
        from copy import deepcopy
        """
        data = self.data.__class__()
        # data.graph_idx = self.data.graph_idx
        # self.data.graph_idx += 1

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            # ADDED: IF STATEMENT
            if key in self.slices.keys():
            
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key,
                        item)] = slice(slices[idx],
                        slices[idx + 1])
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]

        
        #edge_index = data.edge_index
        #node_num = data.x.size()[0]
        #edge_num = data.edge_index.size()[1]
        #data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        
        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)
        """

        data = self.graphs[idx]
        
        if self.aug == 'dnodes':
            data_aug = data_util.drop_nodes(deepcopy(data))
        elif self.aug == 'dedge_nodes':
            data_aug = data_util.drop_edge_nodes(deepcopy(data))
        elif self.aug == 'pedges':
            data_aug = data_util.permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug = data_util.subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug = data_util.mask_nodes(deepcopy(data))
        elif self.aug == 'diff':
            data_aug = data_util.ppr_aug(deepcopy(data))
        elif self.aug == 'rotate':
            data_aug = data_util.rotate(deepcopy(data))
        elif self.aug == 'clip':
            data_aug = data_util.clip(deepcopy(data))
        elif self.aug == 'none':
            data_aug = deepcopy(data)
            self.aug = None
            #data_aug.x = torch.ones((data.edge_index.max()+1, 1))
        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
                data_aug = data_util.drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = data_util.subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random3':
            n = np.random.randint(3)
            if n == 0:
                data_aug = data_util.drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = data_util.permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = data_util.subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = data_util.drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = data_util.permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = data_util.subgraph(deepcopy(data))
            elif n == 3:
                data_aug = data_util.mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False


        if self.weak_aug2 == 'dnodes':
            data_weak_aug2 = data_util.drop_nodes(deepcopy(data))
        elif self.weak_aug2 == 'dedge_nodes':
            data_weak_aug2 = data_util.drop_edge_nodes(deepcopy(data))
        elif self.weak_aug2 == 'pedges':
            data_weak_aug2 = data_util.permute_edges(deepcopy(data))
        elif self.weak_aug2 == 'subgraph':
            data_weak_aug2 = data_util.subgraph(deepcopy(data))
        elif self.weak_aug2 == 'mask_nodes':
            data_weak_aug2 = data_util.mask_nodes(deepcopy(data))
        elif self.aug == 'rotate':
            data_weak_aug2 = data_util.rotate(deepcopy(data))
        elif self.aug == 'clip':
            data_weak_aug2 = data_util.clip(deepcopy(data))
        elif self.weak_aug2 == 'diff':
            data_weak_aug2 = data_util.ppr_aug(deepcopy(data))
        elif self.weak_aug2 == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_weak_aug2 = deepcopy(data)
            self.weak_aug2 = None
            #data_weak_aug2.x = torch.ones((data.edge_index.max()+1, 1))

        # print(self.stro_aug)
        if self.stro_aug == 'stro_subgraph':
            data_stro_aug = data_util.stro_subgraph(deepcopy(data))
        elif self.stro_aug == 'stro_dnodes':
            data_stro_aug = data_util.stro_drop_nodes(deepcopy(data))
        elif self.stro_aug == 'subgraph':
            data_stro_aug = data_util.subgraph(deepcopy(data))
        elif self.stro_aug == None or self.stro_aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_stro_aug = deepcopy(data)
            #data_stro_aug.x = torch.ones((data.edge_index.max()+1, 1))
        else:
            print('stro_subgraph augmentation error')
            assert False

        if self.weak_aug2 != None:
            return data, data_aug, data_weak_aug2
        return data, data_aug, data_stro_aug
        

        

        
"""
def get_adj_matrix(data) -> np.ndarray:
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def ppr_aug(data):
    adj_matrix = get_adj_matrix(data)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix)
    # print(ppr_matrix)

    k = 128
    eps = None

    if k:
        # print(f'Selecting top {k} edges per node.')
        ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps:
        print(f'Selecting edges with weight greater than {eps}.')
        ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError

    edge_index = torch.tensor(ppr_matrix).nonzero().t()
    data.edge_index = edge_index
    # print(data.edge_index)
    return data

def drop_edge_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    ratio = 0.1
    degrees = {}
    for n in data.edge_index[0]:
        if n.item() in degrees:
            degrees[n.item()] += 1
        else:
            degrees[n.item()] = 1
    degrees = sorted(degrees.items(), key = lambda item: item[1])

    idx_drop = np.array([n[0] for n in degrees[:int(len(degrees) * ratio)]])
    drop_num = len(idx_drop)
    drop_ratio = drop_num / node_num
    # print('drop_ratio is: {}'.format(drop_ratio))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    edge_idx_drop = [idx for idx, n in enumerate(data.edge_index[0]) if n.item() in idx_drop or data.edge_index[1][idx].item() in idx_drop]
    edge_idx_nondrop = [n for n in range(len(data.edge_index[0])) if not n in edge_idx_drop]

    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    try:
        data.edge_attr = data.edge_attr[edge_idx_nondrop]
    except:
        pass
    return data

# def flip(x, dim):
#     print(x)
#     indices = [slice(None)] * x.dim()
#     indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
#                                 dtype=torch.long, device=x.device)
#     return x[tuple(indices)]

def rotate(data):
    data.x = torch.flip(data.x, [1])
    data.edge_index = torch.flip(data.edge_index, [1])

    return data


def clip(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num / 10)     # ratio for remained nodes

    # idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_drop = np.arange(drop_num)

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data

def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num / 10)     # ratio for remained nodes
    # print('[info] drop_num is:{}'.format(drop_num))

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    # print('[info] idx_drop is:{}'.format(idx_drop))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data

def stro_drop_nodes(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    degrees = {}
    for n in data.edge_index[0]:
        if n.item() in degrees:
            degrees[n.item()] += 1
        else:
            degrees[n.item()] = 1
    for key, val in degrees.items():
        degrees[key] = math.log(val)

    min_d, max_d = min(degrees.values()), max(degrees.values())

    idx_drop = np.array([n for n in degrees if (degrees[n] - min_d) / (max_d - min_d) <= 0.2])

    drop_num = len(idx_drop)
    drop_ratio = len(idx_drop) / node_num
    # print('drop_ratio is: {}'.format(drop_ratio))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    # print('[info] stro edge_index is:{}'.format(len(edge_index[0])))

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    rand_ids = np.random.choice(edge_num, edge_num-permute_num, replace=False)
    edge_index = edge_index[rand_ids]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    try:
        data.edge_attr = data.edge_attr[edge_idx_nondrop]
    except:
        pass

    return data

# def permute_edges(data):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     permute_num = int(edge_num / 10)
#
#     edge_index = data.edge_index.transpose(0, 1).numpy()
#
#     idx_add = np.random.choice(node_num, (permute_num, 2))
#     # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
#
#     # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
#     # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
#     edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
#     # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
#     data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
#
#     return data

def subgraph(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

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

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def stro_subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.1)
    # print('[info] stro sub_num is:{}'.format(sub_num))
    edge_index = data.edge_index.numpy()

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
    # print('[info] stro idx_drop is:{}'.format(idx_drop))
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    # print('[info] stro idx_sub is:{}'.format(len(idx_sub)))

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_nondrop, :] = 0
    adj[:, idx_nondrop] = 0
    edge_index = adj.nonzero().t()
    # print('[info] stro edge_index is:{}'.format(len(edge_index[0])))

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


"""