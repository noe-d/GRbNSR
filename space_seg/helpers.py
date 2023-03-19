import os

import numpy as np
import seaborn as sns
import time

from synthetic.commands.gen import Gen

# helpers 

"""
Generating Graphs --> from synthetic
"""
generator_cmd = Gen("test")
def gen_from_args(args:dict):
    start_time = time.time()
    generator_cmd.run(args)
    return time.time()-start_time

"""
Graphs number of vertices and edges
"""

def max_possible_edges(nodes):
    return nodes*(nodes-1)/2

def coherent_max_edges(nodes, edges):
    return edges < max_possible_edges(nodes)

def av_deg(nodes, edges):
    return 2*edges/nodes

def sparsity(nodes, edges):
    return edges/max_possible_edges(nodes)

def numbers_of_edges(
    n_nodes = 500
    , densities = [0.01, 0.05, 0.1, 0.5]
):
    n_edges = []
    max_edges = max_possible_edges(n_nodes)
    
    for p in densities:
        n_edges += [int(np.round(p*max_edges))]
        
    
    return n_edges

"""
os and data management
"""

def filter_nonexisting_paths(list_of_paths
                            ):
    non_existing_paths = [p for p in list_of_paths
                          if not os.path.exists(p)]
    
    return non_existing_paths
        
    

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
    return True

def check_meta_args(args_list, req_args):
    availabled_metadata = np.all([ np.all([r in d.keys() for r in req_args]) 
                                  for d in args_list
                                 ]
                                )
    return availabled_metadata

def get_ve_values_form_gen_args(args_list:list # list of dict
                                 , ve_pair:tuple
                                 , keys:list
                                ):
        vc = ve_pair[0]
        ec = ve_pair[1]
        
        values = []
        for k in keys:
            values_k = [d[k] for d in args_list
                        if (d["nodes"]==vc and d["edges"]==ec)
                       ]
            values += [values_k]
        
        return values
    
"""
plotting
"""
    
def remove_frame(ax, remove_annots = True):
    sns.despine(bottom=True, left=True, ax=ax)
    if remove_annots:
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    return