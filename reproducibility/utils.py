"""
         __  _ __    
  __  __/ /_(_) /____
 / / / / __/ / / ___/
/ /_/ / /_/ / (__  ) 
\__,_/\__/_/_/____/  
                     
"""

import os
import pandas as pd
import numpy as np

from synthetic.generator import load_generator

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               Handling files              
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_list_of_nets(folder:str
                     , src_subfolder:str="obs"
                    ):
    
    (_, _, flnms), _ = os.walk(folder+src_subfolder)
    
    return flnms


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               Searching files              
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#               Best run

def optim_value(values:list,
                optim:str="min"
               ):
    if optim=="min": # minimize
        optim_ind = np.argmin(values)
        optim_value = values[optim_ind]
    else: # maximize
        optim_ind = np.argmax(values)
        optim_value = values[optim_ind]
        
    return (optim_value, optim_ind)

def retrieve_best_runs(results_dir:str,
                       n_exps:int=1,
                       experiments_names:list=None,
                       csv_loss_name="evo.csv",
                       loss_col = "best_fit",
                       optim:str="min",
                       include_prog:bool=False,
                      ):
    
    if experiments_names is None:
        experiments_names = []
        res_folders = [x[0] for x in os.walk(results_dir)]# glob("{}/*/".format(results_dir), recursive = True)#
        for f in res_folders:
            if f.endswith(".{}".format(n_exps-1)):
                experiments_names += [f.split("/")[-1].split(".")[0]]
                
    dict_best_runs = {}
    for exp in experiments_names:
        best_fits = []
        for i in range(n_exps):
            # retrieve best fit for curr experiment
            curr_evo = pd.read_csv(results_dir+"/{e}.{ind}/{table}".format(
                e=exp,
                ind=i,
                table=csv_loss_name,
            )
                                  )
            curr_fit, _ = optim_value(values=curr_evo[loss_col], optim=optim)
            best_fits += [curr_fit]
            
        best_run_loss, best_run_id = optim_value(values=best_fits, optim=optim)
            
        dict_best_runs[exp] = {
            "best_run_id":best_run_id,
            "best_run_path":results_dir+"{e}.{ind}/".format(e=exp,ind=best_run_id),
            "best_fit":best_run_loss,
            "n_runs":n_exps,
        }
        
        if include_prog:
            for v in dict_best_runs.values():
                v["prog"] = str(load_generator(v["best_run_path"]+"bestprog.txt", directed=False).prog)
        
    return dict_best_runs

