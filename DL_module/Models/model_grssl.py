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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Graph Representation Self-Supervised Learning
                                                                   (_,_)
"""
from abc import abstractmethod

import torch
import torch.nn.functional as F

import numpy as np

from . import encoders
from .model_util import *
from Trainers.loss import setup_loss_fn

from DataLoader import data_util

from dgl.nn.pytorch import Set2Set

# ====================================================================================
#                         Graph Representation Learning Model
# ====================================================================================

class GRSSLModel(torch.nn.Module):
    def __init__(
        self,
        model_name:str="GRSSLModel",
        encoder_name:str="gin",
        encoder_args:dict={
            'num_layers':4,
            'input_dimension':128,
            'output_dimension':16,
            'hidden_dimension':64,
        },
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.model_type = encoder_name
        self.encoder = encoders.make_encoder(encoder_name
                                             , encoder_arguments=encoder_args
                                            )

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def embed(self,g, **kwargs):
        pass

# ====================================================================================
#                Dummy embedding with SOLELY number of  nodes and edges                                      
# ====================================================================================    

class NodesEdgesModel(torch.nn.Module):
    def __init__(
        self,
        model_name:str="NodesEdgesModel",
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        
    def __str__(
        self,
    ):
        model_info = "NodesEdgesModel(\n\t[# nodes , # edges]\n)"
        
        return model_info
        
    def embed(self, g):
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
            
        return np.array([n_nodes, n_edges])


# ====================================================================================
#                         Embedding with traditional statistics                       
# ====================================================================================

import igraph as ig
import networkx as nx
import dgl

# triad
def triad_count(ig_g):
    motifs = ig_g.motifs_randesu(size=3, cut_prob=None)
    motifs = np.array(motifs)
    
    motifs[np.isnan(motifs)] = 0 
    
    return motifs 

import dgl.function as fn

# PageRank
DAMP = 0.85
K = 10

def compute_pagerank(g):
    N = len(g.nodes())
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_u(u='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']
    

str2stat = {
    "deg": lambda g: np.array(g.in_degrees()),
    "dist": lambda g: np.array([item 
                                for sublist in ig.Graph.from_networkx(nx.to_undirected(dgl.to_networkx(g))).distances(mode=ig.ALL) 
                                for item in sublist 
                                if item > 0
                               ]
                              ),
    "pr": lambda g: compute_pagerank(g),
    "triad": lambda g: triad_count(ig.Graph.from_networkx(nx.to_undirected(dgl.to_networkx(g)))),
    "dummy": lambda g: np.array([1, 1])
}
    
class TradModel(torch.nn.Module):
    def __init__(
        self,
        model_name:str="TradModel",
        stats_dict:dict={
            "deg":{
                "bins":10,
                "range":[0,10],
            },
            "dist":{
                "bins":10,
                "range":[0,10],
            },
            "pr":{
                "bins":10,
                "range":[0,1],
            },
            "triad":{
                "bins":4,
                "range":[0,4],
            },
        },
        dimensionality_reducer = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        
        self.stats_dict = stats_dict
        stats_str = list(self.stats_dict.keys())
        
        self.stats_keys = stats_str
        self.stats_list = []
        
        for k in self.stats_keys:
            k_bins = self.stats_dict[k]["bins"]
            k_range = self.stats_dict[k]["range"] if "range" in self.stats_dict[k].keys() else None
            
            self.init_stat(
                stat_str=k,
                bins = k_bins,
                allowed_range = k_range
            )
            
        self.dim_reducer = dimensionality_reducer # must have a fit_transform(X) method
        self.fitted_dim_reducer = False
        
    def __str__(
        self,
        stat_len:int=8,
        bins_len:int=4,
        range_len:int=8,
    ):
        trad_infos = "TradModel(\n"
        
        for s_name, s_param in self.stats_dict.items():
            name = s_name
            str_bins = str(s_param["bins"] if "bins" in s_param.keys() else "-")
            str_range = str(s_param["range"] if "range" in s_param.keys() else "-")
            
            trad_infos+="\t — {n}{ws_n} ( bins : {b}{ws_b} | range: {r}{ws_r})\n".format(
                n = name,
                b = str_bins,
                r = str_range,
                ws_n = " "*(stat_len-len(name)),
                ws_b = " "*(bins_len-len(str_bins)),
                ws_r = " "*(range_len-len(str_range)),
            )
            
        trad_infos += ")"
        return trad_infos
        
    def init_stat(
        self,
        stat_str:str,
        bins:int=10,
        allowed_range:list=None,
    ):
        ##
        if not stat_str in str2stat.keys():
            raise NotImplementedError("' {given} ' is not one of the implemented methods: {possible}".format(
                given=stat_str,
                possible=str2stat.keys()
            )
                                     )
        ##
        stat = str2stat[stat_str]
        allowed_range = allowed_range
        
        def compute_hist(
            g,
            get_vals,
            bins:int=10,
            allowed_range=None,
        ):
            vals = get_vals(g)
            
            if allowed_range is None:
                allowed_range = [np.min(vals), np.max(vals)]
                
            vals[vals<allowed_range[0]] = allowed_range[0]
            vals[vals>allowed_range[1]] = allowed_range[1]
                
            hist, bins = np.histogram(vals, 
                                      bins=bins,
                                      range=allowed_range
                                     )
            
            return hist
            
        self.stats_list += [
            lambda g: compute_hist(
                g, 
                get_vals = stat,
                bins=bins,
                allowed_range = allowed_range,
            )
        ]
        
        return
    
    def compute_stats(self, g):
        stats_hists = []
        for s in self.stats_list:
            stats_hists += [s(g)]
            
        return stats_hists
    
    def fit_transform_dim_reducer(self, X):
        transformed_X = self.dim_reducer.fit_transform(X)
        self.fitted_dim_reducer = True
        return transformed_X
    
    def train_dim_reducer(self, gs):
        embeddings = np.array([self.stacked_stats(g) for g in gs])
        embeddings = self.dim_reducer.fit(embeddings)
        self.fitted_dim_reducer = True
        
        return self.dim_reducer
        
    def stacked_stats(self, g):
        stacked_stats = self.compute_stats(g)
        stacked_stats = np.array([v for sv in stacked_stats for v in sv])
        return stacked_stats
    
    def embed_list(self, gs):
        embeddings = np.array([self.stacked_stats(g) for g in gs])
        embeddings = self.fit_transform_dim_reducer(embeddings)
        
        return embeddings
    
    def embed(self, g):
        if not self.dim_reducer is None:
            assert self.fitted_dim_reducer, " Dim Reducer not fitted ... "
            
            emb = self.dim_reducer.transform([self.stacked_stats(g)])[0]
        else:
            emb = self.stacked_stats(g)
            
        return emb
    
    
# ====================================================================================
#                                         GCC
# ====================================================================================

class GCCModel(GRSSLModel):
    def __init__(
        self,
        encoder_name:str="gin",
        encoder_args:dict={
            'num_layers':4,
            #'input_dimension':128,
            'output_dimension':16,
            'hidden_dimension':64
        },
        degree_input = True,
        positional_embedding_size=32,
        degree_embedding_size=32,
        max_degree = 128,
        norm=True,

        num_step_set2set=6,
        num_layer_set2set=3,
        **kwargs
    ):

        self.max_degree = max_degree
        self.degree_input = degree_input
        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1

        #encoder_args["input_dimension"] = node_input_dim
        self.rdt_output_dimension = encoder_args["output_dimension"]
        transmitted_encoder_args = {}
        transmitted_encoder_args.update(encoder_args)
        transmitted_encoder_args["input_dimension"] = node_input_dim
        transmitted_encoder_args["output_dimension"] = encoder_args["hidden_dimension"] # output comes at the lin_readout here

        super().__init__(
            model_name="GCC",
            encoder_name=encoder_name,
            encoder_args=transmitted_encoder_args,#encoder_args,
        )

        if self.degree_input:
            self.degree_embedding = torch.nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )

        self.set2set = Set2Set(self.encoder.hidden_dimension
                               , num_step_set2set
                               , num_layer_set2set
                              )
        self.lin_readout = torch.nn.Sequential(
            torch.nn.Linear(2 * self.encoder.hidden_dimension, self.encoder.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder.hidden_dimension, self.rdt_output_dimension),
        )
        self.norm = norm


    def forward(self, g):
        """Predict molecule labels
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : Predicted labels
        """

        # nfreq = g.ndata["nfreq"]
        if self.degree_input:
            device = g.ndata["seed"].device
            degrees = g.in_degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    # g.ndata["pos_directed"],
                    # self.node_freq_embedding(nfreq.clamp(0, self.max_node_freq)),
                    # self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                    # nfreq.unsqueeze(1).float() / self.max_node_freq,
                    # degrees.unsqueeze(1).float() / self.max_degree,
                ),
                dim=-1,
            )

        # efreq = g.edata["efreq"]
        # e_feat = torch.cat(
        #     (
        #         self.edge_freq_embedding(efreq.clamp(0, self.max_edge_freq)),
        #         efreq.unsqueeze(1).float() / self.max_edge_freq,
        #     ),
        #     dim=-1,
        # )

        e_feat = None
        if self.model_type == "gin":
            x = self.encoder(g, n_feat)
        else:
            x = self.encoder(g, n_feat)
            #print(g.num_nodes(), n_feat.shape, x.shape)
            x = self.set2set(g, x)
            x = self.lin_readout(x)
        if self.norm:
            x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-5)

        return x

    def embed(self, g, **kwargs):

        pos_emb_size = self.encoder.input_dimension-(self.degree_embedding.embedding_dim+1)

        g = data_util._add_undirected_graph_positional_embedding(g, hidden_size=pos_emb_size)
        degs = g.in_degrees()
        # as if use_entire_graph == True:
        g.ndata["seed"] = torch.zeros(g.number_of_nodes(), dtype=torch.long)
        g.ndata["seed"][degs.argmax().item()] = 1

        try:
            _device = torch.cuda.current_device()
        except:
            _device = torch.device("cpu")

        #print(g.ndata["pos_undirected"].shape, self.degree_embedding(degs.to(torch.cuda.current_device())).shape, g.ndata["seed"].shape)

        n_feat = torch.cat(
            (
                g.ndata["pos_undirected"].to(_device),
                self.degree_embedding(degs.to(_device).clamp(0, self.max_degree)),
                g.ndata["seed"].to(_device).unsqueeze(1).float(),
            ),
            dim=-1,
        ).to(_device)

        #print(n_feat.shape)

        g = g.to(_device)

        if self.model_type == "gin":
            x = self.encoder(g, n_feat)
        else:
            x = self.encoder(g, n_feat)
            #print(g.num_nodes(), n_feat.shape, x.shape)
            x = self.set2set(g, x)
            x = self.lin_readout(x)
        if self.norm:
            x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-5)

        emb = np.array(x.detach()[0])

        """
        from DataLoader.data_loader import GCCDataset

        # make dataset w/ 1. graph
        solo_data = GCCDataset(dgl_graphs=[g], dataset_name="EMBED SOLO", **kwargs)
        # --> get graph => augment graph + contrast
        graph_q, graph_k = solo_data[0]
        # forward into the model
        with torch.no_grad():
            feat_q = self.forward(graph_q.to(torch.cuda.current_device()))
            #feat_k = self.forward(graph_k.to(torch.cuda.current_device()))
            feat_k = feat_q
        # avg. for embedding
        emb = (feat_q + feat_k) / 2
        """

        return emb

# ====================================================================================
#                                       GraphMAE
# ====================================================================================

class GraphMAEModel(GRSSLModel):
    def __init__(
        self,
        encoder_name:str="gin",
        encoder_args:dict={},
        decoder_name:str="gin",
        mask_rate: float = 0.3,
        replace_rate: float = 0.1,
        drop_edge_rate: float = 0.0,
        concat_hidden: bool = False,
        #alpha_l: float = 2,
        #loss_fn: str = "sce",
        **kwargs
    ):
        #encoder_args["output_dimension"] = encoder_args["hidden_dimension"]

        super().__init__(
            model_name="GraphMAE",
            encoder_name=encoder_name,
            encoder_args=encoder_args
        )

        decoder_args = {}
        if encoder_name == decoder_name:
            decoder_args = encoder_args.copy()
        decoder_args["input_dimension"]=self.encoder.output_dimension
        # decoder_args["hidden_dimension"]=self.encoder.hidden_dimension // self.encoder.num_heads if "gat" in decoder_name else self.encoder.hidden_dimension
        if not "gat" in decoder_name:
            decoder_args["hidden_dimension"]=self.encoder.hidden_dimension
        decoder_args["output_dimension"]=self.encoder.input_dimension
        decoder_args["num_layers"] = self.encoder.num_layers

        self.decoder = GRSSLModel(encoder_name=decoder_name,
                                  encoder_args=decoder_args
                                 )

        self._mask_rate = mask_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._drop_edge_rate = drop_edge_rate
        self.enc_mask_token = torch.nn.Parameter(torch.zeros(1, self.encoder.input_dimension))

        self._concat_hidden = concat_hidden
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(self.decoder.encoder.input_dimension * self.encoder.num_layers
                                                , self.decoder.encoder.input_dimension
                                                , bias=False
                                               )
        else:
            self.encoder_to_decoder = nn.Linear(self.decoder.encoder.input_dimension
                                                , self.decoder.encoder.input_dimension
                                                , bias=False
                                               )

        # * setup loss function
        #loss_args = {"alpha":alpha_l}
        #self.criterion = setup_loss_fn(loss_fn,
        #                               alpha=alpha_l
        #                              )

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        #loss = self.mask_attr_prediction(g, x)
        x_rec, x_init = self.mask_attr_prediction(g, x)

        #loss_item = {"loss": loss.item()}
        return x_rec, x_init#loss, loss_item




    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, g, x, return_mask=False):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self.decoder.model_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self.decoder.model_type in ("mlp", "linear") :
            recon = self.decoder.encoder(rep)
        else:
            recon = self.decoder.encoder(pre_use_g, rep)


        if return_mask:
            return x, recon, mask_nodes
        else:
            x_init = x[mask_nodes]
            x_rec = recon[mask_nodes]

            #loss = self.criterion(x_rec, x_init)
            return x_init, x_rec

    def embed(self
              , g
              , pooler:str = "sum"
             ): # TODO: [ ] load data to GPU instead of unloading model to cpu. ....
        if not "attr" in g.ndata:
            degs = g.in_degrees()
            
            deg_roofed = degs
            MAX_DEGREES=self.encoder.input_dimension-1
            deg_roofed[deg_roofed>MAX_DEGREES] = MAX_DEGREES
            degs = deg_roofed
            
            feats = F.one_hot(
                degs, 
                num_classes=self.encoder.input_dimension
            ).float()

            g.ndata["attr"] = feats
            
        pooler_method = str2readout(pooler)
            
        emb = self.encoder.to('cpu').forward(g.to('cpu'), g.ndata["attr"].to('cpu'))
        emb = np.array(pooler_method(g, emb).detach()[0])

        return emb

    # ====================================================
    # ⚠️ Deprecated versions

    def forward_deprecated(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction_deprecated(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self.decoder.model_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self.decoder.model_type in ("mlp", "linear") :
            recon = self.decoder.encoder(rep)
        else:
            recon = self.decoder.encoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    

# ====================================================================================
#                                         PGCL
# ====================================================================================

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class PGCLModel(GRSSLModel):
    def __init__(
        self,
        
        # SUPER
        encoder_name:str="gin",
        encoder_args:dict={},
        
        # PGCL
        nmb_prototypes=0,
        alpha=0.5,
        beta=1.,
        gamma=.1,
        
        prior:bool=True,
        
        **kwargs
    ):
        #encoder_args["output_dimension"] = encoder_args["hidden_dimension"]

        super().__init__(
            model_name="PGCL",
            encoder_name=encoder_name,
            encoder_args=encoder_args
        )
        
        self.nmb_prototypes=nmb_prototypes
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        
        self.prior=prior
        
        """
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.output_dimension, self.encoder.output_dimension),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.encoder.output_dimension, self.encoder.output_dimension)
        )
        """
        
        # prototype layer
        self.prototypes = None
        proto_size = self.encoder.output_dimension#*self.encoder.num_gc_layers
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(proto_size, self.nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(proto_size, self.nmb_prototypes, bias=False)

            
        self.local_d = FF(self.encoder.output_dimension)
        self.global_d = FF(self.encoder.output_dimension)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.encoder.output_dimension)
            
        #self.criterion = torch.nn.CrossEntropyLoss()
        
    """    
    def forward(self, g, x=None, do_unbatch = True):
        if do_unbatch:
            batch_graphs = dgl.unbatch(g)
            
            embs, outs = [], []
            for bg in batch_graphs:
                emb, out = self.forward(bg, do_unbatch=False)
                embs += [torch.Tensor(emb)]
                outs += [torch.Tensor(out)]
                
            return torch.cat(embs), torch.cat(outs)
            
        else:
        
            if not "attr" in g.ndata:
                degs = g.in_degrees()

                deg_roofed = degs
                MAX_DEGREES=self.encoder.input_dimension-1
                deg_roofed[deg_roofed>MAX_DEGREES] = MAX_DEGREES
                degs = deg_roofed

                feats = F.one_hot(
                    degs, 
                    num_classes=self.encoder.input_dimension
                ).float()

                g.ndata["attr"] = feats
                #g.ndata["attr"] = torch.ones((g.num_nodes(), 1))
        
            x=g.ndata['attr']

            h = self.encoder(g
                             #, x
                             #, return_hidden=False
                            )

            if self.prototypes is not None:
                return h, self.prototypes(h)
            else:
                return h
    """
    def forward(self, g):
                #, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        #if x is None:
        #    x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(g)#x, edge_index, batch)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)

        if self.prototypes is not None:
            return y, self.prototypes(y)
        else:
            return y
            
            
    def embed(self
              , g
              #, pooler:str = "avg"
             ):
        with torch.no_grad():
            x, emb = self.encoder.forward(g)
            
            emb = emb[0].clone().detach()
            
        return np.array(emb)
        #if self.prototypes is not None:
        #    return np.array(emb[0][0].detach())
        #else:
        #    return np.array(emb[0].detach())
        
        
    def forward_OG(self
                , x
                , edge_index
                , batch
                , num_graphs
               ):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)

        if self.prototypes is not None:
            return y, self.prototypes(y)
        else:
            return y