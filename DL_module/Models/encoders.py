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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Encoders
                                                                   (_,_)
TODOs:

- [X] GAT
- [x] GIN
- [x] GCN

"""

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import (
    GATConv,
    GATv2Conv,
    GraphConv,
    GINConv,
)
from dgl.utils import expand_as_pair
import dgl.function as fn

from .model_util import *

# ====================================================================================
#                         Graph Representation Learning Model                        
# ====================================================================================

def make_encoder(encoder_name:str="gin",
                 encoder_arguments:dict={
                     'num_layers':4,
                     'input_dimension':128,
                     'output_dimension':16,
                     'hidden_dimension':64
                 },
                )->nn.Module:
    """
    Instantiate the layers of the Graph (Neural Network) model (encoder /(opt) decoder)
    based on a model type (eg. GAT, GIN, GCN, ...) 
    """
    
    if encoder_name=="gat":
        return GAT(**encoder_arguments)
    elif encoder_name=="gin":
        return GIN(**encoder_arguments)
    elif encoder_name=="gcn":
        return GCN(**encoder_arguments)
    elif encoder_name=="pgcl":
        return PGCLEncoder(**encoder_arguments)
    else:
        raise NotImplementedError("/!\ '{e}' type is not recognized.\nPlease use one of the defined layer type: {types}.\nAlternatively, define it in '{f}'.".format(e=encoder_name,types=['gat', 'gin', 'gcn'], f=__file__))
    
    print("🚧 WIP: instantiating {m} to output representations of {d} dimensions".format(m=encoder_name, d=output_dimension))
    
    return encoder_name

# ====================================================================================
#                                   Encoding modules                                  
# ====================================================================================

class Encoder(nn.Module):
    def __init__(
        self,
        layer_component:nn.Module,
        num_layers:int,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int,
        activations_list:list=[None],
        **kwargs
    ):
        # 1. asserts
        
        # 2. super
        super().__init__()
        
        # 3. instantiate 
        self.num_layers = num_layers
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        
        if not len(activations_list)==self.num_layers:
            activations_list = [activations_list[0]]*self.num_layers
        self.activations_list = activations_list
            
        self.layers = nn.ModuleList(
            [
                layer_component(
                    in_feats=self.input_dimension if i==0 else self.hidden_dimension,
                    out_feats=self.hidden_dimension if i<self.num_layers-1 else self.output_dimension,
                    **kwargs,                    
                    activation = str2act(self.activations_list[i])
                )
                for i in range(self.num_layers)
            ]
        )
        
        
    def __add__(self,
                encoder2,
               ):
        # instantiate resulting Encoder
        resulting_encoder = copy.deepcopy(self)
        # assertions / raise errors
        if self.output_dimension != encoder2.input_dimension:
            raise ValueError("1st Encoder's output dimensions ({o1}) does not match 2nd Encoder's input dimension ({i2}).".format(o1=self.output_dimension, i2=encoder2.input_dimension))
        # resulting input dimension: same as self (copied)
        resulting_encoder.output_dimension = encoder2.output_dimension
        
        resulting_encoder.num_layers += encoder2.num_layers
        resulting_encoder.layers = nn.ModuleList([l for l in self.layers]+[l for l in encoder2.layers])
                
        return resulting_encoder
        
        
    #def forward():
     #   pass
    
# ====================================================================================
#                                         GAT                                                                          
# ====================================================================================

dict_str2agg = {
    "mean":torch.mean,
    "sum":torch.sum,
}

class GAT(Encoder):
    def __init__(
        self,
        num_layers:int,
        input_dimension:int,
        output_dimension:int,
        num_heads:int,
        activation:str='none',
        v2:bool=False,
        agg_mode:str="mean",
        **kwargs
    ):
        if v2:
            gat_layer = GATv2Conv
        else:
            gat_layer = GATConv
            
        activations_list = ['leakyrelu' for _ in range(num_layers-1)]+['none']
        
        # based on: out_feats = node_hidden_dim // num_heads
        hidden_dimension = int(output_dimension*num_heads)
        kwargs["num_heads"]=num_heads
        kwargs["allow_zero_in_degree"] = True

        super().__init__(
            layer_component=gat_layer,
            
            num_layers=num_layers,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            activations_list=activations_list,
            
            **kwargs
        )
        
        self.num_heads = num_heads
        
        if not agg_mode in dict_str2agg.keys():
            raise NotImplementedError("Aggregation of attention heads must be one of: {l}.\n{in_mode} is not.".format(l=dict_str2agg.keys, in_mode=agg_mode))
        else:
            self.agg_method = dict_str2agg[agg_mode]
        
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_reps = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            #h = h.flatten(1)
            h = self.agg_method(h, dim=1)
            hidden_reps += [h]
            
        if return_hidden:
            return h, hidden_reps
        else:
            return h

# ====================================================================================
#                                         GCN                                         
# ====================================================================================

class GCN(Encoder):
    def __init__(
        self,
        num_layers:int,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int,
        activation:str='relu',
        dropout: float = 0.,
        readout = None,
        readout_args = {},
        layernorm:bool = False,
        **kwargs
    ):
        gcn_layer = GraphConv
        activations_list = [activation for _ in range(num_layers-1)]+['none']
        kwargs["allow_zero_in_degree"] = True
        
        super().__init__(
            layer_component=gcn_layer,
            
            num_layers=num_layers,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            activations_list=activations_list,
            
            **kwargs
        )
        
        self.dropout = dropout
        
        if readout=="set2set":
            readout_args["input_dim"]=self.hidden_dimension
            self.linear = nn.Linear(2 * self.hidden_dimension, self.hidden_dimension)
        self.readout=str2readout(readout, readout_args)
        
        layernorm = nn.LayerNorm(self.hidden_dimension, elementwise_affine=False) if layernorm else None
        self.ln = layernorm

        
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_reps = [h]
        
        for i, layer in enumerate(self.layers):
            if self.dropout is not None and self.dropout>0.:
                h = F.dropout(h, p=self.dropout)#, training=self.training)
                
            h = layer(g, h)
            hidden_reps += [h]
            
        if self.readout is not None:
            h = self.readout(g, h)
            if isinstance(self.readout, Set2Set):
                h = self.linear(h)
                
        if self.ln is not None:
            h = self.ln(h)
            
        if return_hidden:
            return h, hidden_reps
        else:
            return h
        
    """
    def GCCforward(self, g, feats, efeats=None):
        for layer in self.layers:
            feats = layer(g, feats)
        feats = self.readout(g, feats)
        if isinstance(self.readout, Set2Set):
            feats = self.linear(feats)
        if self.layernorm:
            feats = self.ln(feats)
        return feats
    
    
    
    def GraphMAEforward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers: # norms is always None !!
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)        
    """

# ====================================================================================
#                                         GIN                                         
# MLP + GIN
# ====================================================================================

"""
class GIN(Encoder):
    
    # (1) MLP layer as input of the apply_func
    
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
        use_selayer,
        
       V in_dim,
       V num_hidden,
       V out_dim,
       V num_layers,
       — dropout,
       X activation,
       X residual,
       X norm,
       X encoding=False,
       X learn_eps=False,
       — aggr="sum",
        
    ):
"""

class GIN(Encoder):
    def __init__(
        self,
        num_layers:int,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int,
        activation:str='none',
        
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        norm='none',
        residual=False,
        num_mlp_layers:int=2,
        mlp_hidden_dimension:int=None,
        
        dropout: float = 0.,
        final_dropout: float = 0.,
        graph_pooling_type:str="root",
        
        **kwargs
    ):
        gin_layer = GINLayer
        activations_list = [activation for _ in range(num_layers-1)]+['none']
                
        kwargs["aggregator_type"] = aggregator_type
        kwargs["init_eps"] = init_eps
        kwargs["learn_eps"] = learn_eps
        kwargs["residual"] = residual
        kwargs["num_mlp_layers"] = num_mlp_layers
        kwargs["mlp_hidden_dimension"] = mlp_hidden_dimension
        kwargs["norm"] = norm
                
        super().__init__(
            layer_component=gin_layer,
            
            num_layers=num_layers,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            activations_list=activations_list,
            
            **kwargs
        )
        
        self.dropout = dropout
        
        self.graph_pooling_type = graph_pooling_type
        self.graph_pooling = str2readout(graph_pooling_type)
        
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dimension, output_dimension))
            else:#elif layer < (self.num_layers -1):
                self.linears_prediction.append(nn.Linear(hidden_dimension, output_dimension))
            #else:
            #    self.linears_prediction.append(nn.Linear(hidden_dimension, output_dimension))

        self.final_drop = nn.Dropout(final_dropout)
    
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_reps = [h]
        for i, layer in enumerate(self.layers):
            if self.dropout >0.:
                h = F.dropout(h, p=self.dropout)
            h = layer(g, h)
            hidden_reps += [h]
        
        if not self.graph_pooling_type in ([None, 'none']):
            score_over_layer = 0
            for i, h in enumerate(hidden_reps[:-1]):
                pooled_h = self.graph_pooling(g, h)
                score_over_layer += self.final_drop(self.linears_prediction[i](pooled_h))
                
            
            returned = score_over_layer
        else:
            returned = h
            
        if return_hidden:
            return returned, hidden_reps
        else:
            return returned
    
#                                        Classes 
# - [x] MLP
# - [x] ApplyFunc 
# - [x] GINLayer



class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels),
            nn.ELU(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU.

    Merge from GCC & GraphMAE
    """
    def __init__(
        self,
        mlp,
        norm:str="batchnorm", # GraphMAE --> merge with GCC: use_selayer [bool]
        activation:str="relu", # GraphMAE
    ):
        super().__init__()
        
        self.mlp = mlp
        self.norm = (
            SELayer(self.mlp.output_dimension
                    , int(np.sqrt(self.mlp.output_dimension))
                   )
            if norm=="use_selayer"
            else str2norm(norm)
        )
        self.act = str2act(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h
    
        
class PLayer(nn.Linear):
    def __init__(
        self,
        in_feats,
        out_feats,
        activation,
        norm:str='none',
        bias:bool=True,
        device=None,
        dtype=None
    ):
        super().__init__(
            in_features=in_feats,
            out_features=out_feats,
            bias = bias,
            device=device,
            dtype=dtype
        )
        
        self.act = activation
        self.norm = str2norm(norm)
            
    def forward(self, x, last_layer:bool=True):
        if last_layer:
            return super().forward(x)
        else:
            return self.act(self.norm(super().forward(x)))
    
class MLP(Encoder):
    def __init__(
        self,
        num_layers:int,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int,
        activation:str="relu",
        **kwargs
    ):
        mlp_layer = PLayer

        activations = [activation for _ in range(num_layers-1)]+['none']
        
        super().__init__(
            layer_component=mlp_layer,
            num_layers=num_layers,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            activations_list=activations,
            **kwargs
        )
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            #print(i, h)
            #print(layer.act, layer.norm)
            last_layer = i == (self.num_layers-1)
            h = layer(h, last_layer=last_layer)
        return h

class GINLayer(nn.Module):
    """
    Rewritten from GINCov
    
    """
    def __init__(
        self,
        in_feats,
        out_feats,
        #apply_func=ApplyNodeFunc,
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        norm=None,
        activation=None,
        residual=False,
        
        num_mlp_layers:int=2,
        mlp_hidden_dimension:int=None,
    ):
        super().__init__()
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        
        #self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self._reducer = str2aggr(aggregator_type)
        
        self.activation = activation
        self.norm=norm
        self.residual = residual
        
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
        
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
            
        self.mlp = MLP(num_layers=num_mlp_layers,
                       input_dimension=in_feats,
                       output_dimension=out_feats,
                       hidden_dimension=mlp_hidden_dimension if mlp_hidden_dimension is not None else out_feats,
                       activation=self.activation,
                       **{"norm":self.norm,
                         }
                      )
        
        self.apply_func = ApplyNodeFunc(self.mlp,
                                        norm=self.norm,
                                        activation=self.activation,
                                       )
        
    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # --> activation already in apply_func

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst
        
        
"""
PGCL TRIALS
"""

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
import dgl

class PGCLEncoder(torch.nn.Module):
    def __init__(
        self
        , num_features
        , dim
        , num_gc_layers
    ):
        super().__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_features = num_features
        self.num_gc_layers = num_gc_layers
        
        self.hidden_dimension = dim
        self.output_dimension = int(self.hidden_dimension*self.num_gc_layers)
        #self.hidden_dimension = self.output_dimension

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.output_dimension, self.output_dimension),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.output_dimension, self.output_dimension))
        
        for i in range(self.num_gc_layers):
            if i:
                nn = Sequential(Linear(self.hidden_dimension, self.hidden_dimension)
                                , ReLU()
                                , Linear(self.hidden_dimension, self.hidden_dimension)
                               )
            else:
                nn = Sequential(Linear(self.num_features, self.hidden_dimension)
                                , ReLU()
                                , Linear(self.hidden_dimension, self.hidden_dimension)
                               )
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(self.hidden_dimension)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, g):
                #, x, edge_index, batch):
        x, edge_index, batch, _ = get_x_edges_batch_ng(g)
        #if x is None:
        #    x = 
        #x = torch.ones((g.num_nodes(), 1))#.to(device)
        #edge_index = g.edges()
        #edge_index = torch.stack(list(edge_index), dim=0)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2

        #xpool = [global_add_pool(x
        #                         ,None# batch
        #                        )
        #         for x in xs]
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        # x = F.dropout(x, p=0.5, training=self.training)
        y = self.proj_head(x)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        # y = F.dropout(y, p=0.5, training=self.training)
        return y, x


    

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                try:
                    data = data[0]
                except TypeError:
                    pass
                data.to(device)
                #x, edge_index, batch = data.x, data.edge_index, data.batch
                x, edge_index, batch, _ = get_x_edges_batch_ng(data)
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                # x, _ = self.forward(x, edge_index, batch)
                x, emb = self.forward(data)#x, edge_index, batch)

                # ret.append(x.cpu().numpy())
                ret.append(emb.cpu().numpy())
                #y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        #y = np.concatenate(y, 0)
        return ret#, y

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                    break

        return x_g, ret, y