import numpy as np

import sys
sys.path.append('./DL_module') # to be fixed ?? 
from DL_module.Models import from_pretrained as pretrained
from dgl import from_networkx as nx2dgl

from synthetic.distances import *
from synthetic.net import Net
from dgl import DGLGraph

from scipy.spatial.distance import cosine, euclidean

def str2dist(strdist):
    if strdist=="cosine":
        return cosine
    elif strdist=="euclidean":
        return euclidean
    else:
        raise NotImplementedError

class DeepDistancesToNet(object):
    def __init__(
        self, 
        net,
        model=None,
        embed_args:dict={},
        embedder=None,
        model_name:str=None,
        dist_type:str="cosine",
        norm=Norm.ER_MEAN_RATIO,
        norm_samples=DEFAULT_NORM_SAMPLES,
    ):
        assert(norm == Norm.NONE or norm_samples > 0)
        assert((not model is None) or (not embedder is None))

        self.net = net
        self.dgl_net = nx2dgl(net.graph.to_networkx().to_undirected())
        
        # TODO : load model
        if embedder is None:
            model = pretrained.get_model(model)
            self.embed = pretrained.represent_from_model(model, **embed_args)
            del model
        else:
            self.embed = embedder
        
        self.net_emb = self._init_emb()
        
        # TODO : load distance
        self.dist_type = dist_type # cosine, euclidian, manhattan, ...
        self.dist = str2dist(self.dist_type)

        self.norm = norm #if not use_netlsd_dist else None
        if norm != Norm.NONE:
            self.norm_values = self._compute_norm_values(net, norm_samples)
        else:
            self.norm_values = None
            
            
    def _init_emb(self):
        self.net_emb = self.embed(self.dgl_net) # TODO: handle format and conversion if necessary
            
        return self.net_emb
            
    def _compute_norm_values(self, net, norm_samples):
        start_time = current_time_millis()

        norm_values = []
        dists2net = DeepDistancesToNet(
            net,
            embedder = self.embed,
            dist_type = self.dist_type,
            norm=Norm.NONE,
        )
        i = 0
        print('computing normalization samples...')
        with progressbar.ProgressBar(max_value=norm_samples) as bar:
            for i in range(norm_samples):
                bar.update(i)
                
                vcount = net.graph.vcount()#.number_of_nodes()#net.graph.vcount()
                ecount = net.graph.ecount()#number_of_edges()//2#net.graph.ecount()
                
                sample_net = create_random_net(vcount, ecount, False)
                
                #sample_net = nx2dgl(sample_net.graph.to_networkx())
                
                dist = dists2net.compute(sample_net)
                
                norm_values += [dist]
                
        norm_values = [np.mean(norm_values)]

        comp_time = (current_time_millis() - start_time) / 1000
        print('{}s'.format(comp_time))

        return norm_values
    
    def compute(self, net):
        
        targ_emb = self.embed(nx2dgl(net.graph.to_networkx().to_undirected())) # TODO: handle format and conversion if necessary
        
        distance = self.dist(self.net_emb, targ_emb)
        
        if self.norm == Norm.ER_MEAN_RATIO:
            distance /= self.norm_values[0]

        return [distance]
    
    " ==================================================== "
    " > Attempt to resolve internally the type of network  "
    "          initially given to the DeepDistToNet class. "
    " ==================================================== "
    
    def compute_dgl(self, targ_dgl):
        targ_emb = self.embed(targ_dgl)
        distance = self.dist(self.net_emb, targ_emb)
        
        if self.norm == Norm.ER_MEAN_RATIO:
            distance /= self.norm_values[0]
            
        return distance
    
    def _init_compute_method(self, net=None):
        if net is None:
            net = self.net
        if np.any([subc==Net for subc in net.__class__.__mro__]):
            self.compute_method = lambda targ_net: self.compute_dgl(
                targ_dgl = nx2dgl(targ_net.graph.to_networkx().to_undirected())
            )
            
        elif np.any([subc==DGLGraph for subc in net.__class__.__mro__]):
            self.compute_method = lambda targ_net: self.compute_dgl(targ_net)

        else:
            raise NotImplementedError
            
        return
        