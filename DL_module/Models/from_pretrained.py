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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _) Make Loading pre-trained models eaasy
                                                                   (_,_)
"""

import os
import torch

from Configs.configs_parser import ConfigParser
import Models.model_grssl as module_arch
import Trainers.trainer as module_trainer
import DataLoader.data_loader as module_dataset

from Models.model_util import str2readout
from Utils.misc import read_json


# ====================================================================================
#                                    Loading Model                                   
# ====================================================================================

def get_model(input_model):
    """
    handle various types of input_models:
        - class Model --> pass
        - folder (with files: config.json and model_best.pth) --> load
        - class Trainer --> extract model
    """
    # 1. retrieve the model from different formats
    # 1.1 if given model IS a GRSSLModel: all good
    if module_arch.GRSSLModel in input_model.__class__.__mro__:
        model = input_model 
    # 1.2 if given is <str> --> config folder to be loaded
    elif type(input_model) == str:
        if os.path.exists(input_model+"/model_best.pth"):
            model = model_from_checkpoint(path_to_folder=input_model,
                                          # other default args
                                         )
        elif input_model.endswith('.pth'): # new
            path_to_folder = "/".join(input_model.split("/")[:-1])
            model_name = input_model.split('/')[-1]
            path_to_config = path_to_folder+"/config.json"
            path_to_model = path_to_folder+"/{}".format(model_name)
            model = model_from_checkpoint(
                path_to_config=path_to_config,
                path_to_model = path_to_model
            )
        else:
            try:
                json_file = input_model if input_model.endswith('.json') else input_model+'config.json'
                json_config = read_json(json_file)
                
                model_name = json_config["arch"]["type"]
                model_args = json_config["arch"]["args"]
                model = getattr(module_arch, model_name)(**model_args)
            except:
                raise NotImplementedError("Unable to load model from: ' {} '".format(input_model))
                
    # 1.3 if given is <Trainer> instance: extract model
    #                      and in this case:
    #                                 + dataset if not other given
    #                                 + logger if not other given
    elif module_trainer.Trainer in input_model.__class__.__mro__:
            
        model = input_model.model
    # 1.4 else... ERROR
    else:
        raise NotImplementedError("<model> format not recognize for:\n{}.".format(input_model))
        
    return model

def model_from_checkpoint(
    path_to_folder:str=None,
    model_name:str="model_best.pth",
    config_name:str="config.json",
    path_to_config:str=None,
    path_to_model:str=None,
):
    """
    load the model from config and checkpoint files
    """
    ##
    err_asrt_msg = "One of <path_to_folder> or <path_to_config> and <path_to_model> should not be `None`"
    assert (not path_to_folder is None) or ((not path_to_config is None) and (not path_to_model is None)), err_asrt_msg
    ##
    
    if path_to_config is None:
        path_to_config = path_to_folder + config_name
    if path_to_model is None:
        path_to_model = path_to_folder + model_name


    config = ConfigParser.from_json(json_path = path_to_config
                                    , run_id=''
                                    , write_config=False
                                   )
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(path_to_model)
    if checkpoint['config']['arch'] != config['arch']:
            print("Warning: Architecture configuration given in config file is different from that of "
                  "checkpoint. This may yield an exception while state_dict is being loaded.")

    success = model.load_state_dict(checkpoint['state_dict'])
    
    ##
    assert success, "Error when loading model's state dict..."
    ##
    
    # freeze weights
    model.eval()
    
    return model


# ====================================================================================
#                                      Representer                                     
# ====================================================================================

def represent_from_model(
    model,
    #pooler=None, #"avg" --> to be directly encompassed in <embed> method ??
    **kwargs,
):
    
    #representer = lambda g: model.embed(g)
    #if not pooler is None:
    #    representer = lambda g: model.embed(g, pooler)
    #else:
    #    representer = lambda g: model.embed(g)
        
    representer = lambda g: model.embed(g, **kwargs)
    
    return representer

def representer_from_checkpoint(
    path_to_folder:str=None,                       
    load_model_args:dict={},
    representer_args:dict={},
):
    model = model_from_checkpoint(
        path_to_folder=path_to_folder,
        **load_model_args
    )
    
    representer = represent_from_model(
        model = model,
        **representer_args
    )
    
    return representer



# ====================================================================================
#                                    Loading Dataset                                   
# ====================================================================================

def get_dataset(input_dataset):
    """
    load dataset regardless of the format. Could be:
        - 
    """
    if module_dataset.GraphDataset in input_dataset.__class__.__mro__:
        dataset = input_dataset
    # 2.2 load model from str
    elif type(input_dataset) == str:
        try:
            dataset = module_dataset.GraphDataset(
                dgl_graphs=input_dataset,
                verbosity=False,
                dataset_name=input_dataset,
            )
        except:
            raise NotImplementedError("Unable to load dataset from give str key: '{}'.".format(input_dataset))
    elif module_trainer.Trainer in input_dataset.__class__.__mro__:
        dataset = input_dataset.data_loader.dataset
    else:
        raise NotImplementedError("<dataset> format not recognize for:\n{}.".format(dataset))
        
    return dataset