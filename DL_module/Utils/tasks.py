"""
,---------.    ____       .-'''-. .--.   .--.     .-'''-.  
\          \ .'  __ `.   / _     \|  | _/  /     / _     \ 
 `--.  ,---'/   '  \  \ (`' )/`--'| (`' ) /     (`' )/`--' 
    |   \   |___|  /  |(_ o _).   |(_ ()_)     (_ o _).    
    :_ _:      _.-`   | (_,_). '. | (_,_)   __  (_,_). '.  
    (_I_)   .'   _    |.---.  \  :|  |\ \  |  |.---.  \  : 
   (_(=)_)  |  _( )_  |\    `-'  ||  | \ `'   /\    `-'  | 
    (_I_)   \ (_ o _) / \       / |  |  \    /  \       /    _ 
    '---'    '.(_,_).'   `-...-'  `--'   `'-'    `-...-'   _( )_   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _) 
                                                           (_,_)
   _ 
 _( )_   
(_ o _)  Graph Classification
 (_,_)

"""

from time import time

import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

import torch

from dgl.data import TUDataset

from .misc import set_seed, seconds2hms

import DataLoader.data_loader as module_dataset
import Trainers.trainer as module_trainer
import Models.model_grssl as module_arch
from Models.model_util import str2readout
import train

import Models.from_pretrained as pretrained

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_TU_NAMES = [ # non-exhaustive list
    "REDDIT-BINARY"
    , "IMDB-BINARY"
    , "IMDB-MULTI"
    , "DD"
    , "ENZYMES"
    , "COLLAB"
    , "PROTEINS"
    , "REDDIT-MULTI-5K"
]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ====================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Graph Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ====================================================================================

# DUMMY BASELINES  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_IMPLEMENTED_DUMMIES = ["most_common"
                        , "shuffle_random"
                       ]
_SKLEARN_DUMMIES = [
    "most_frequent"
    , "prior"
    , "stratified"
    , "uniform"
]

def get_dummy_baseline(x,
                       y,
                       dummy_type,
                       seed:int=0,
                       n:int=10
                      ):
    if not dummy_type in _SKLEARN_DUMMIES:
        raise KeyError
        
    else:
        kf = StratifiedKFold(n_splits=n
                             , shuffle=True
                             , random_state=seed
                            )
        scores = []
        for train_index, test_index in kf.split(x, y):
            clf = DummyClassifier(strategy=dummy_type)

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            
            clf.fit(x_train, y_train)
            scores += [clf.score(x_test, y_test)]
            
        dummy_dict = {"{} - micro-F1".format(dummy_type): np.mean(scores),
                      "{} - micro-F1 (std.)".format(dummy_type): np.std(scores)
                     }
    
    return dummy_dict

def most_common_classify(labels, n:int=10):
    most_common_lab = max(set(labels), key = labels.count)
    most_common_preds = [most_common_lab]*len(labels)
    
    #most_common_acc = accuracy_score(labels, most_common_preds)
    most_common_f1 = f1_score(labels, most_common_preds, average="micro")
    
    return {#"most_common accuracy":most_common_acc,
            "most_common micro F1":most_common_f1
           }

def rdm_shuffle_classify(labels, n:int=10):
    accuracies = []
    
    for _ in range(n):
        shuffled_labels = labels.copy()
        np.random.shuffle(shuffled_labels)
        accuracies += [accuracy_score(labels, shuffled_labels)]
    
    return {"shuffle micro F1 (av.)": np.mean(accuracies),
            "shuffle micro F1 (std.)": np.std(accuracies)
           }
    

# CLASSIFICATION METHOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def svc_classify(x, y
                 , search:bool=True
                 , seed:int=0
                 , nfolds:int=10
                 , do_shuffle:bool=True
                ):
    kf = StratifiedKFold(n_splits=nfolds
                         , shuffle=True
                         , random_state=seed
                        )
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {"C": [1, 10, 100, 1000, 10000, 100000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
            )
        else:
            classifier = SVC(C=100000)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        
        #print(classifier.best_params_)
        
    return {"Micro-F1": np.mean(accuracies),
            "Micro-F1 (std.)": np.std(accuracies)
           }


# CLASS: Graph Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GraphClassification():
    def __init__(
        self,
        dataset,
        model=None,
        pooler=None,
        provided_embeddings=None,
        seed:int=0,
        n_repeat:int=10,
        dummy_baselines:list=["most_frequent","uniform"]
    ):
        assert not (((model is None) or (pooler is None)) and (provided_embeddings is None)), "A model, pooler pair OR pre-computed embeddings must be provided to GraphClassification"
        
        self.graphs = dataset.graphs
        self.labels = dataset.graphs_labels
        
        self._seed = seed
        self._repeat = n_repeat
        
        from tqdm import tqdm
        
        if provided_embeddings is None:
            model.eval()
            #embeddings = [pooler(g, model.embed(g, g.ndata["attr"])) for g in self.graphs]
            embeddings = [pooler(g, model.embed(g)) for g in tqdm(self.graphs)]
            self.embeddings = torch.cat([emb.detach().cpu() for emb in embeddings])
        else:
            assert len(provided_embeddings)==len(self.labels), "Pre-computed provided embeddings length ({l_embs}) must match dataset lentgh ({l_data}).".format(l_embs=len(provided_embeddings), l_data=len(self.labels))
            self.embeddings = provided_embeddings
        #embeddings = [pooler(g, model(g[0])) for g in dataset]
        
        
        self.dummy_baselines = dummy_baselines
        
    def classification_scores(self,
                              **kwargs
                             ):
        
        scores = svc_classify(
            x = self.embeddings,
            y = self.labels,
            seed = self._seed,
            nfolds = self._repeat,
            **kwargs
        )
        
        if hasattr(self, "scores"):
            self.scores.update(scores)
        else:
            self.scores = scores
        
        return scores
    
    def get_dummy_preds(self):
        dummy_preds = {}
        
        for dummy in self.dummy_baselines:
            dummy_preds.update(get_dummy_baseline(x=self.embeddings
                                                  , y=self.labels
                                                  , dummy_type=dummy
                                                  , seed=self._seed
                                                  , n = self._repeat
                                                 )
                              )
        
        
        if hasattr(self, "scores"):
            self.scores.update(dummy_preds)
        else:
            self.scores = dummy_preds
        
        return dummy_preds
    
    def print_scores(self, do_print:bool=True):
        s_str = "Scores:\n"
        s_str += "\n".join(["\t- {n} = {s:.4g}".format(n=k,s=v) for k, v in self.scores.items()])
        
        if do_print:
            print(s_str)
            return
        else:
            return s_str
        
    def write_results_csv(self
                          , path_to_csv:str
                          , data_name:str
                          , run_id:str
                          , score_key:str="Micro-F1"
                         ):
        try:
            results_df = pd.read_csv(path_to_csv, header=0, index_col=0)
        except:
            results_df = pd.DataFrame()
            
        results_df.loc[data_name, str(run_id)] = self.scores[score_key]
        
        results_df.to_csv(path_to_csv)
        
        return
    
    
# Wrap-up:

def testmodel_dataset(
    model,
    dataset=None,
    embedding_args:dict = {},
    #pooler_type:str="avg",
    #pooler_args:dict={},
    logger=None,
    seed:int=0,
    n_repeat:int=10,
    dummy_baselines:list=["most_frequent","uniform"],
):
    """
    # =============================================
    # 1. retrieve the model from different formats
    # 1.1 if given model IS a GRSSLModel: all good
    if module_arch.GRSSLModel in model.__class__.__mro__:
        pass
    # 1.2 if given is <str> --> config folder to be loaded
    elif type(model) == str:
        model = pretrained.model_from_checkpoint(path_to_folder=model,
                                                 # other default args
                                                )
    # 1.3 if given is <Trainer> instance: extract model
    #                      and in this case:
    #                                 + dataset if not other given
    #                                 + logger if not other given
    elif module_trainer.Trainer in model.__class__.__mro__:
        if dataset is None:
            dataset = model.data_loader.dataset
        if logger is None:
            logger = model.logger
            
        model = model.model
    # 1.4 else... ERROR
    else:
        raise NotImplementedError("<model> format not recognize for:\n{}.".format(model))
        
    if logger is None: # make logger: print if not given
        logger = print
    else:
        logger = logger.info

    logger(model)
    # =============================================
    # 2. retrieve the dataset from its given name
    # 2.1 if given dataset IS already a dataset: all good
    if module_dataset.GraphDataset in dataset.__class__.__mro__:
        pass
    # 2.2 load model from str
    elif type(dataset) == str:
        # 2.2-1 if TU dataset --> load and give to GraphDataset
        if dataset in _TU_NAMES:
            tu_data = TUDataset(dataset)
            
            tu_graphs = tu_data.graph_lists
            tu_labels = [int(lab[0]) for lab in tu_data.graph_labels]
            
            dataset = module_dataset.GraphDataset(
                dgl_graphs=tu_graphs,
                graphs_labels=tu_labels,
                og_format="dgl",
                verbosity=False,
                dataset_name=dataset,
            )
            
        # 2.2-2 if .csv --> load Netzschleuder like dataset
        elif dataset.endswith(".csv") or (dataset.split(" ")[0].endswith(".csv")):
            dataset = module_dataset.GraphDataset(
                dgl_graphs=dataset,
                graphs_labels=None,
                og_format="dgl",
                verbosity=False,
                dataset_name=dataset,
                #display_col_width=60,
                from_csv=True, # ??
                name_col="Name",
                label_col="Category",
            )
        else:
            raise NotImplementedError("Unable to load dataset from give str key: '{}'.".format(dataset))
    else:
        raise NotImplementedError("<dataset> format not recognize for:\n{}.".format(dataset))
        
    logger(dataset)
    """
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
    # 3. Compute embeddings on the dataset's graphs
    display_logger("Computing embeddings .... ")
    model.eval()
    emb_time = time()
    dataset_embeddings = np.array([
        model.embed(g, **embedding_args) for g in data.graphs
    ])
    emb_time = time()-emb_time
    display_logger("\t... embeddings computed in: {}.".format(seconds2hms(emb_time)))
    
    # =============================================
    # 4. Graph Classification 
    # 4.1 instantiation w/ pre-computed embeddings
    test_classification = GraphClassification(
        dataset=data,
        model=model,
        #pooler=pooler, --> deprecated
        provided_embeddings=dataset_embeddings,
        
        seed=seed,
        n_repeat=n_repeat,
        dummy_baselines=dummy_baselines,
    )
    
    del data
    
    # 4.2 Classification & scores computation
    # 4.2-1 SVC based on embeddings
    display_logger("{}-folds SVC-Classification from embeddings ...".format(n_repeat))
    classification_time = time()
    test_classification.classification_scores()
    classification_time = time()-classification_time
    display_logger("\t... classification computed in: {}.".format(seconds2hms(classification_time)))
    
    # 4.2-2 dummy baseline
    display_logger("Computing dummy predictions: {}.".format(dummy_baselines))
    test_classification.get_dummy_preds()
    
    # 4.2-3 display scores
    display_logger(test_classification.print_scores(do_print=False))
    
    
    return test_classification
        
""" DEPRECATED
def testmodel_classification(config
                             , dataset_name:None
                             , pooler_type:str="avg", pooler_args:dict={}
                             , seed:int=0
                             , n_repeat:int=10
                             , dummy_baselines:list=["most_frequent","uniform"]
                            ):
    config_to_load = deepcopy(config)
    
    # alter config
    config_to_load._config["do_train"] = False

    if (dataset_name is None) or dataset_name == config["dataset"]["args"]["dgl_graphs"]:
        pass#data_to_test = trainer_loaded.data_loader.dataset
    elif dataset_name in _TU_NAMES:
        tu_data = TUDataset(dataset_name)
        
        tu_graphs = tu_data.graph_lists
        tu_labels = [int(lab[0]) for lab in tu_data.graph_labels]
        
        config_to_load._config["dataset"]["args"]["from_csv"] = False
        config_to_load._config["dataset"]["args"]["dataset_name"] = dataset_name
        config_to_load._config["dataset"]["args"]["dgl_graphs"] = tu_graphs
        config_to_load._config["dataset"]["args"]["graphs_labels"] = tu_labels
        
        #data_to_test = config_to_load.init_obj('dataset', module_dataset)
    else:
        raise NotImplementedError("Error testing {} dataset".format(dataset_name))
        
    # load trainer
    trainer_loaded = train.main(config_to_load)
    
    model_to_test = trainer_loaded.model
    data_to_test = trainer_loaded.data_loader.dataset
    
    pooler = str2readout(pooler_type, **pooler_args)
    
    trainer_loaded.logger.info("Computing embeddings ...")
    emb_time = time()
    test_classification = GraphClassification(
        dataset=data_to_test,
        model=model_to_test,
        pooler=pooler,
        seed=seed,
        n_repeat=n_repeat,
        dummy_baselines=dummy_baselines
    )
    emb_time = time()-emb_time
    trainer_loaded.logger.info("... embeddings computed in: {}.".format(seconds2hms(emb_time)))
    
    trainer_loaded.logger.info("{}-folds SVC-Classification from embeddings ...".format(n_repeat))
    classification_time = time()
    test_classification.classification_scores()
    classification_time = time()-classification_time
    trainer_loaded.logger.info("... classification computed in: {}.".format(seconds2hms(classification_time)))
    
    trainer_loaded.logger.info("Computing dummy predictions: {}.".format(dummy_baselines))
    test_classification.get_dummy_preds()
    
    trainer_loaded.logger.info("{}".format(test_classification.print_scores(do_print=False)))
    
    return test_classification
"""