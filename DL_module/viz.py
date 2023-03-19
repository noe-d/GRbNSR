"""
         __
 __  __ /\_\  ____    
/\ \/\ \\/\ \/\_ ,`\  
\ \ \_/ |\ \ \/_/  /_ 
 \ \___/  \ \_\/\____\
  \/__/    \/_/\/____/
  <———————————————————>
   
Getting visualisation of the Graph Representations given a model and a dataset

To run this script, use the following command line:

```
python viz.py --c CONFIG_FILE_PATH
```

Note:
    - dataset : can be overwritten using 
        `--dataset_name DATASET_NAME` (eg. with a TUDataset, for instance "REDDIT-BINARY")
    - reducer : can be overwritten using 
        `--reducer REDUCER_NAME` (name from sklearn eg: "manifold.TSNE")
    - interactive: can be overwritten using
        `--no_interactive` (hinders to plot the interactive plot w/ plotly)
"""
import argparse
from Configs.configs_parser import ConfigParser

from Utils.misc import *#prepare_device, set_seed

import Visualisers.visualiser as module_viz
import train

from copy import deepcopy
import torch
import pandas as pd
from time import time

from Models.model_util import str2readout
import Models.from_pretrained as pretrained
import Trainers.trainer as module_trainer

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_DEFAULT_CONFIG = "Configs/config_files/config_graphmae_vGCN.json"
_DEFAULT_PATH_MODEL = "./saved/models/GraphMAE_GCNversion/0125_114556/"

_LOGGER_NAME = "visualisation"

_DEFAULT_STATIC_ARGS = {
    "color_attr": "label",
    "cmap": "viridis",
    "fig_dimensions": (10,8),
    "color_list": module_viz.cm.get_cmap("tab10").colors,
    "categorical_color": True,
    #"save_path":"illustrations/rdt_bin_tsne_trad_degs.png",
}
_DEFAULT_INTERACT_ARGS = {
    "min_size":500,
    "color_attr":"label",
}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def main(
    model,
    dataset=None,
    model_name=None,
    embedding_args:dict={},
    pooler_args:dict={},
    red_type:str="manifold.TSNE",#None
    red_args:dict={},
    no_interactive:bool=True,
    save_path:str=None,
):
    # =============================================
    # 1. Load components
    # 1.1 load model
    model_loaded = pretrained.get_model(model)
    # 1.2 load dataset
    if not dataset is None:
        data_loaded = pretrained.get_dataset(dataset)
    else: # if Trainer instance was given ...
        data_loaded = pretrained.get_dataset(model)
        
    # (1.3) get logger ?
    if module_trainer.Trainer in model.__class__.__mro__:
        display_logger = model.logger.info
    else:
        display_logger = print
        
    # put together (1.1&1.2)
    model = model_loaded
    data = data_loaded
    
    del model_loaded, data_loaded, dataset
    
    display_logger(model)
    display_logger(data)
    # =============================================
    # 2. Compute embeddings on the dataset's graphs
    display_logger("Computing embeddings .... ")
    model.eval()
    emb_time = time()
    computed_embeddings = np.array([
        model.embed(g, **embedding_args) for g in data.graphs
    ])
    emb_time = time()-emb_time
    display_logger("\t... embeddings computed in: {}.".format(seconds2hms(emb_time)))
    
    del model
    
    # =============================================
    # 3. Prepare visualisation
    # 3.1 TODO: [ ] Set displayable mode for the graphs depending on their OG source
    if data._og_format == "graph-tool":
        #csv_path, max_n = config["dataset"]["args"]["dgl_graphs"].split(" ")
        csv_path = data.name
        whole_graphs_data = pd.read_csv(csv_path)
        
        if hasattr(data, "non_loaded_graphs"):
            mask_loaded = np.array([
                row["Name"] not in data.non_loaded_graphs
                for _, row in whole_graphs_data.iterrows()
            ])
        else:
            max_n = len(data)
            mask_loaded = np.array([
                i < max_n
                for i, _ in whole_graphs_data.iterrows()
            ])
        
        graphs_data_df = whole_graphs_data[mask_loaded]#pd.read_csv(csv_path)
        graphs_names = graphs_data_df["Name"]#[:int(max_n)]
        graphs_info = graphs_data_df["Tags"]#[:int(max_n)]
        
        graphs_to_visualise = graphs_names

    else:
        graphs_to_visualise = data.graphs
        graphs_names = [i for i, _ in enumerate(graphs_to_visualise)]
        graphs_info = [None for _ in graphs_names]
        
    og_graph_format = data._og_format
    data_name = data.name
    str_labels = [data.labId_to_labName[lab] for lab in data.graphs_labels]
    
    del data
    
    # 3.2 init GRVisualiser
    display_logger("📽 Initialising the Graph Representation Visualiser")
    if not red_type is None:
        print("projecting embeddings with {}.".format(red_type.split(".")[-1]))
    visualiser = module_viz.GRVisualiser(
        graphs = graphs_to_visualise, 
        embeddings = computed_embeddings,
        labels = str_labels,
        og_format = og_graph_format,
        
        reducer_type = red_type,
        reducer_args = red_args,
        
        make_interactive = not no_interactive, # TODO: parse
        dataset_name = data_name,# if data.name is not None else "",
        model_name = model_name, # todo retrieve model name
    )
    
    # =============================================
    # 4. ✨  visualise  ✨
    static_args = _DEFAULT_STATIC_ARGS
    if not save_path is None:
        static_args.update(
            {"save_path":save_path}
        )
    interactive_args = _DEFAULT_INTERACT_ARGS
    
    plots = visualiser.show(
        static_args = static_args,
        interactive_args = interactive_args
    )
    
    if not no_interactive:
        app = plots[1]
        
        app_mode = "external"
        app_debug = True
        app_host = '0.0.0.0'#"127.0.0.1"
        app_port = "8060"
        
        app.run_server(app_mode
                       , debug=app_debug
                       , port=app_port
                       , host=app_host
                       , use_reloader=False
                      )
    
    return
    

# ====================================================================================
#                                       __main__                                                                              
# ====================================================================================

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Model and Graphs dataset to be visualised.')
    #args.add_argument('-c'
    #                  , '--config'
    #                  , default=_DEFAULT_CONFIG
    #                  , type=str
    #                  , help='config file path (default: None)'
    #                 )
    args.add_argument('-m'
                      , '--model'
                      , default=_DEFAULT_PATH_MODEL
                      , type=str
                      , help='model folder path'
                     )
    args.add_argument('-d'
                      , '--dataset'
                      , default=None
                      , type=str
                      , help='dataset to visualise'
                     )
    args.add_argument('-p'
                      , '--pooler'
                      , default="sum"
                      , type=str
                      , help='Pooling layer for Graph Representation'
                     )
    args.add_argument('-r'
                      , '--reducer'
                      , default="manifold.TSNE"
                      , type=str
                      , help='dimensionality reduction (from sklearn) (default: TSNE)'
                     )
    args.add_argument('-s'
                      , '--save_path'
                      , default=None
                      , type=str
                      , help='path to save fig'
                     )
    args.add_argument('-f'
                      , '--force_save'
                      , action='store_true'
                      , help='force to save the figure in `illustrations/` folder'
                     )
    args.add_argument('-i'
                      , '--interactive'
                      #, default=None
                      , action='store_false'
                      , help='output the interactive plot'
                     )
    args = args.parse_args()
    
    if args.force_save and (args.save_path is None):
        model_name = args.model.split("/")[-2]
        data_name = args.dataset
        
        save_path = "illustrations/emb_viz_{model}_{data}".format(
            model=model_name,
            data=data_name
        )
        
    else:
        save_path = args.save_path
    
    # modify some args if necessary
    ## ensure no training mode

    
    main(
        model=args.model,
        dataset=args.dataset,
        model_name=None,
        embedding_args={}, # TODO !!
        pooler_args={}, # TODO !!
        red_type=args.reducer,#None
        red_args={}, # TODO !!
        no_interactive=args.interactive,
        save_path = save_path
    )
    
    """
    main(config
         , pooler_type=args.pooler # TODO: depending on the model type
         , pooler_args={} #TODO
         , red_type=args.reducer
         , red_args={} #TODO
         , dataset_name=args.dataset
         , no_interactive=args.no_interactive
        )
    """
    
    
    
    
    
"""
def main_deprecated(config
         , pooler_type="avg"
         , pooler_args:dict={}
         , red_type:str="manifold.TSNE"#None
         , red_args:dict={}
         , dataset_name:str=None
         , no_interactive:bool=True
        ):
    config_to_load = deepcopy(config)
    logger = config_to_load.get_logger(_LOGGER_NAME)
    #config_to_load._config["do_train"] = False
    
    if dataset_name is not None:
        logger.info("🧳 Setting visualisation dataset to '{}'.".format(dataset_name))
        config_to_load._config["dataset"]["args"]["dgl_graphs"] = dataset_name
        
    # load model / trainer
    trainer_loaded = train.main(config_to_load)
    model = trainer_loaded.model
    data = trainer_loaded.data_loader.dataset
    
    gnn_name = config_to_load["name"]
    og_graph_format = data._og_format
    
    # generatet embeddings
    logger.info("🤿 Computing {n} graphs embeddings with: {m} ....".format(n=len(data),m=gnn_name))
    emb_time = time()
    model.eval()
    with torch.no_grad():
        computed_embeddings = [model.embed(g) for g in data.graphs]
        #TODO: parse / determine
        print(computed_embeddings[0].shape)
        size_emb0 = computed_embeddings[0]
        requires_pooler = False#len(computed_embeddings[0].size())>1 and computed_embeddings[0].size(dim=0)>1
        if requires_pooler:
            if pooler_type=="root": # if no given pooler or wrong pooler --> default: "avg"
                pooler_type = "avg"
            pooler = str2readout(pooler_type, **pooler_args)
            computed_embeddings = [pooler(g, e) for g, e in zip(data.graphs, computed_embeddings)]

        computed_embeddings = torch.cat([emb.detach().cpu() for emb in computed_embeddings])
        
    emb_time = time()-emb_time
    logger.info("🌊 ... embeddings computed in {time}!".format(time=seconds2hms(emb_time)))
    
    # TODO : [ ] graphs in displayable format depending on OG source
    if data._og_format == "graph-tool":
        csv_path, max_n = config["dataset"]["args"]["dgl_graphs"].split(" ")
        graphs_data_df = pd.read_csv(csv_path)
        graphs_names = graphs_data_df["Name"][:int(max_n)]
        graphs_info = graphs_data_df["Tags"][:int(max_n)]
        
        graphs_to_visualise = graphs_names

    else:
        graphs_to_visualise = data.graphs
        graphs_names = [i for i, _ in enumerate(graphs_to_visualise)]
        graphs_info = [None for _ in graphs_names]
        
    str_labels = [data.labId_to_labName[lab] for lab in data.graphs_labels]
    
    del trainer_loaded, model, data
    
    # interective ?
    #no_interactive = True

    # init GRVisualiser
    logger.info("📽 Initialising the Graph Representation Visualiser and projecting embeddings with {}.".format(red_type.split(".")[-1]))
    visualiser = module_viz.GRVisualiser(
        graphs = graphs_to_visualise, 
        embeddings = computed_embeddings,
        labels = str_labels,
        og_format = og_graph_format,
        
        reducer_type = red_type,
        reducer_args = red_args,
        
        make_interactive = not no_interactive, # TODO: parse
        dataset_name = dataset_name if dataset_name is not None else "",
        model_name = gnn_name,
        
    )
    
    # ✨  visualise  ✨
    plots = visualiser.show(
        static_args = _DEFAULT_STATIC_ARGS,
        interactive_args = _DEFAULT_INTERACT_ARGS
    )
    
    if not no_interactive:
        app = plots[1]
        
        app_mode = "external"
        app_debug = True
        app_host = '0.0.0.0'#"127.0.0.1"
        app_port = "8060"
        
        app.run_server(app_mode
                       , debug=app_debug
                       , port=app_port
                       , host=app_host
                      )
    
    return

"""