import numpy as np

def emb_degs(net, **kwargs):
    return np.histogram(net.degree(net.graph.vs.indices), **kwargs)[0]

"""
================================================================================================
================================================================================================
================================================================================================
================================================================================================
"""

import sys
from copy import deepcopy

sys.path.insert(1, './DL_module/') # 📦 branch to desired module 
import Models.from_pretrained as pretrained
from train import *
from Models.model_util import str2readout

import dgl
import torch.nn.functional as F

def load_embedder(model:str):
    
    model = pretrained.get_model(model)
    representer = pretrained.represent_from_model(model)
    del model
    
    embedder = representer
    
    return embedder
"""
def load_embedder(config_path="./DL_module/Configs/config_files/"
                 ):
    
    config = ConfigParser.from_json(json_path = config_path+"/config.json"
                                    , run_id = ""
                                   )
    
    config_to_load = deepcopy(config)
    config_to_load._config["do_train"] = False
    config_to_load._config["name"] = config_to_load._config["name"]+config_path.split("/")[-1]
    
    config_to_load._config["dataset"]["args"]["dgl_graphs"] = [dgl.graph(([0], [1]))]
    
    config_to_load._config["trainer"]["save_dir"] = config_to_load._config["trainer"]["save_dir"]
    
    trainer_loaded = main(config_to_load)
    trainer_loaded.model.eval()
    
    # GraphMAE model requires a pooler, while GCC implementation would directly output a vector representation
    pooler = str2readout("avg") 
    #pooler = str2readout("root")
    
    embedder_method = lambda g: pooler(g, trainer_loaded.model.embed(g))
    
    return embedder_method
"""

def embed_deep(net, embedder_method, **kwargs):
    
    g = dgl.from_networkx(net.graph.to_networkx())
    #degs = g.in_degrees()
    #feats = F.one_hot(degs, num_classes=512).float()
    
    #g.ndata["attr"] = feats
    
    return embedder_method(g, **kwargs)#.detach().numpy()[0]

