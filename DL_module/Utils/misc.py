"""
,---.    ,---..-./`)    .-'''-.     _______    
|    \  /    |\ .-.')  / _     \   /   __  \   
|  ,  \/  ,  |/ `-' \ (`' )/`--'  | ,_/  \__)  
|  |\_   /|  | `-'`"`(_ o _).   ,-./  )        
|  _( )_/ |  | .---.  (_,_). '. \  '_ '`)      
| (_ o _) |  | |   | .---.  \  : > (_)  )  __  
|  (_,_)  |  | |   | \    `-'  |(  .  .-'_/  ) 
|  |      |  | |   |  \       /  `-'`-'     /  
'--'      '--' '---'   `-...-'     `._____.'   
                                               
"""

import json
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

def seconds2hms(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec) 

# pytorch-template

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
        
def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def set_seed(SEED:int=0):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    return