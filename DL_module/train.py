"""
 __                                  
/\ \__                 __            
\ \ ,_\  _ __    __   /\_\    ___    
 \ \ \/ /\`'__\/'__`\ \/\ \ /' _ `\  
  \ \ \_\ \ \//\ \L\.\_\ \ \/\ \/\ \ 
   \ \__\\ \_\\ \__/.\_\\ \_\ \_\ \_\
    \/__/ \/_/ \/__/\/_/ \/_/\/_/\/_/
  <———————————————————————————————————>
   
Pipeline module chaining the different steps to train a
Graph Representation Self-Supervised Learning (GRSSL) model.

To run this script, use the following command line:

```
python train.py --c CONFIG_FILE_PATH
```

"""

import argparse
from time import time
from os import listdir
import numpy as np

import DataLoader.data_loader as module_dataset
import Models.model_grssl as module_arch
import Trainers.loss as module_loss
import Trainers.trainer as module_train

from Configs.configs_parser import ConfigParser, _update_config
from Utils.misc import *#prepare_device, set_seed

import torch
from dgl.dataloading import GraphDataLoader

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_LOGGER_NAME = "train"
_DEFAULT_CONFIG = "Configs/config_files/config_graphmae_repro.json"
_DEFAULT_SEED = None
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main(config):
    # set seed for reproducibility
    set_seed(config["seed"])
    
    # instantiate logger
    logger = config.get_logger(_LOGGER_NAME)

    # 1.1 load dataset 
    logger.info("🌐 Loading Dataset...")
    dataset = config.init_obj('dataset', module_dataset)
    logger.info("Dataset loaded:\n{}".format(dataset))

    # 1.2 setup data_loader instances
    data_loader = dataset.make_GraphDataLoader(config["data_loader"])
    valid_data_loader = None#data_loader.split_validation()

    # 2.1 Instantiate Model architecture
    model = config.init_obj('arch', module_arch)
    logger.info("🤖 Model instantiated:{name} ({l})\n{m}".format(name=model.model_name
                                                                 , l=model.model_type
                                                                 , m=model
                                                                )
               )
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 2.2 get function handles of loss and metrics
    criterion = config.init_obj('loss_type', module_loss)
    metrics = None#[getattr(module_metric, met) for met in config['metrics']]

    # 2.3 build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    # 3.1 build trainer 
    trainer_type = getattr(module_train, config['trainer_type']["type"])
    trainer = trainer_type(model,
                           criterion,
                           metrics,
                           optimizer,
                           config=config,
                           device=device,
                           data_loader=data_loader,
                           valid_data_loader=valid_data_loader,
                           lr_scheduler=lr_scheduler,
                           **config['trainer_type']['args']
                          )
    # 3.2.A 🧞 train the model ...
    if "do_train" not in config._config.keys():
        config._config["do_train"] = True # by default: train the model
        
    if config["do_train"]:
        logger.info("⏳ Training Model .....")
        training_time = time()
        trainer.train()
        training_time = time() - training_time

        logger.info("⌛️ Training done in {time}!\nThe trained model is saved at: {path}".format(
            path=trainer.checkpoint_dir,
            time=seconds2hms(training_time)
        )
                   )
    # 3.2.B ... or load "model_best" 
    else:
        saved_files = listdir("."/trainer.checkpoint_dir)
        
        if "model_best.pth" in saved_files:
            model_path =  trainer.checkpoint_dir / "model_best.pth"
            logger.info("⏳ Loading Best Model at {} .....".format(trainer.checkpoint_dir))
        elif np.any(["checkpoint-epoch" in f for f in saved_files]):
            saved_chkpoints = [f for f in saved_files if "checkpoint-epoch" in f]
            saved_chkpoints = [int(chkpt.split("checkpoint-epoch")[1].split(".")[0]) for chkpt in saved_chkpoints]
            model_path = trainer.checkpoint_dir / "checkpoint-epoch{}.pth".format(np.max(saved_chkpoints))
            logger.info("⏳ Loading Last Model at {} .....".format(trainer.checkpoint_dir))
        else:
            raise NotImplementedError("Unable to load model from directory: {} .".format(trainer.checkpoint_dir))
            
        
        trainer._resume_checkpoint(model_path)
        
        #logger.info("⏳ Loading Model from checkpoint at {} .....".format(model_path))
        #checkpoint = torch.load(model_path)
        #state_dict = checkpoint['state_dict']
        #trainer.model.load_state_dict(state_dict, strict=False)
        logger.info("⌛️ ..... Model :{}: loaded.".format(model_path.name))

    
    return trainer
        
    
# ====================================================================================
#                                       __main__                                                                              
# ====================================================================================
    
if __name__ == "__main__":
    # get the config file path
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c'
                      , '--config'
                      , default=_DEFAULT_CONFIG
                      , type=str
                      , help='config file path (default: {})'.format(_DEFAULT_CONFIG)
                     )
    args.add_argument('-s'
                      , '--seed'
                      , default=_DEFAULT_SEED
                      , type=int
                      , help='seed for reproducibility (default: {})'.format(_DEFAULT_SEED)
                     )
    args = args.parse_args()
    # parse the config file
    #CONFIG_PATH = "Configs/config.json"
    config = ConfigParser.from_json(
        json_path = args.config
        , run_id = str(args.seed)
    )
    
    if not args.seed is None:
        config._config["seed"] = args.seed
    
    # --> Call the main module
    # 1. Load Dataset & make data loader
    # 2. Instantiate trainer
    # 3. Train Model
    main(config)