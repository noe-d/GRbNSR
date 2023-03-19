"""
    _______      ,-----.    ,---.   .--. ________ .-./`)   .-_'''-.    
   /   __  \   .'  .-,  '.  |    \  |  ||        |\ .-.') '_( )_   \   
  | ,_/  \__) / ,-.|  \ _ \ |  ,  \ |  ||   .----'/ `-' \|(_ o _)|  '  
,-./  )      ;  \  '_ /  | :|  |\_ \|  ||  _|____  `-'`"`. (_,_)/___|  
\  '_ '`)    |  _`,/ \ _/  ||  _( )_\  ||_( )_   | .---. |  |  .-----. 
 > (_)  )  __: (  '\_/ \   ;| (_ o _)  |(_ o._)__| |   | '  \  '-   .' 
(  .  .-'_/  )\ `"/  \  ) / |  (_,_)\  ||(_,_)     |   |  \  `-'`   |  
 `-'`-'     /  '. \_/``".'  |  |    |  ||   |      |   |   \        /    _
   `._____.'     '-----'    '--'    '--''---'      '---'    `'-...-'   _( )_ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Inspect parsed configs
                                                                       (_,_)
"""

_IMPLEMENTED_TYPES = ["GCC", "GraphMAE"]

def inspect_config(config):
    model_type = config["arch"]["type"]
    dataset_type = config["dataset"]["type"]
    trainer_type = config["trainer_type"]["type"]
    
    # check that types are coherent and implemented
    config_type = check_coherent_types(model_type, dataset_type, trainer_type)
    
    # check that all required arguments are there
    
    
    # check argument go together well
    if config_type == "GraphMAE": # /!\ Not True !!!
        max_deg = config["dataset"]["args"]["MAX_DEGREES"]
        input_size = config["arch"]["args"]["encoder_args"]["input_dimension"]
        assert max_deg==(input_size-1), "The maximum degree ({md}) and input size ({ins}) are not consistent.".format(md=max_deg, ins=input_size)
        
    if config_type=="GCC":
        model_input_size = config["arch"]["args"]["encoder_args"]["input_dimension"]
        model_hidden_size = config["arch"]["args"]["encoder_args"]["hidden_dimension"]
        model_output_size = config["arch"]["args"]["encoder_args"]["output_dimension"]
        
        model_deg_emb_size = config["arch"]["args"]["degree_embedding_size"] if config["arch"]["args"]["degree_input"] else 0
        model_pos_emb_size = config["arch"]["args"]["positional_embedding_size"]
        
        data_subg_size = config["dataset"]["args"]["subgraph_size"]
        data_pos_emb_size = config["dataset"]["args"]["positional_embedding_size"]
        
        
        hidd_embs_check = (model_deg_emb_size + data_pos_emb_size + 1) == model_hidden_size
        
        #assert hidd_embs_check, "Ooopsie"
        
    
    
    return

# ===========================================

def check_coherent_types(model_type:str
                         , dataset_type:str
                         , trainer_type:str
                        ):
    mt = model_type[:-5]
    dt = dataset_type[:-7]
    tt = trainer_type[:-7]
    
    if not len(set([mt, dt, tt])) == 1:
        raise KeyError("Types of dataset ({d}), model ({m}) and trainer ({t}) must be coherent.".format(d=dt, m=mt, t=tt))
    elif not mt in _IMPLEMENTED_TYPES:
        raise NotImplementedError("Type {} not implemented.".format(mt))
    
    return mt

# ===========================================
# GCC

def check_gcc_dims():
    return
    