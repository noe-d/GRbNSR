import copy
import pandas as pd

import subprocess
import multiprocessing as mp

from .helpers import *
from .helpers_emb import *
from .helpers_visu import *
from .helpers_stats import *

from synthetic.net import load_net

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_GRAPHS = 64
GEN_PATH = "space_seg/generators/"
DATA_PATH = "space_seg/data/data_v2/"

N_PROCESSES = 2
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class SynNetsGenerator(object):
    def __init__(
        self,
        generator_dict:dict,
        ve_pairs_list:list,
        
        undir:bool=True,
        
        data_dir:str=DATA_PATH,
        extension:str="pickle",
        
        n_processes:int=N_PROCESSES,
        
        verbosity:int=0,
        compute_files:bool=True,
        do_overwrite:bool=False,
        check_load:bool=False,
    ):
        self.gen_dict = copy.deepcopy(generator_dict)
        self.ve_pairs = ve_pairs_list
        
        self.gen_ext = extension
        self.base_net_paths = data_dir+"v{vc}_e{ec}/{fam_name}/{subfam_name}_{ind}.{ext}"
        
        self.verbose = verbosity > 0
        
        self.gen_args_list = [] # to be filled on init_files
        self.undir = undir
        
        self.n_processes = n_processes
        
        self._overwrite = do_overwrite
        self._check_load = check_load
        files_paths = self._init()
        if compute_files:
            if not self._overwrite:
                # generate all files that cannot be loaded (including existing but empty!)
                if self._check_load:
                    to_be_gen_files = []
                    for f in files_paths:
                        try:
                            load_net(f, directed=False)
                        except:
                            to_be_gen_files += [f]
                # generate only files that do not exist
                else:
                    to_be_gen_files = filter_nonexisting_paths(files_paths)
                        
                if len(to_be_gen_files) > 0:
                    self._init_files(files_paths)
                    times = self._generate_nets(files_paths=to_be_gen_files)
                    self.computed_times = times
                    
            else:
                self._init_files(files_paths)
                times = self._generate_nets()
                self.computed_times = times
        
    def _init(self
             ):
        # retrieve each branching and leaf
        files_paths = []
        for fam_name, fam_dict in self.gen_dict.items():
            for ve in self.ve_pairs:
                
                curr_vc = ve[0]
                curr_ec = ve[1]
                
                curr_files = []

                curr_ind = 0
                for gen_name, gen_dict in fam_dict.items():
                    
                    n_gens = gen_dict["number_generated"]

                    curr_gen_files = []

                    for i in range(curr_ind+1, curr_ind+n_gens+1):
                        
                        curr_onet = self.base_net_paths.format(
                            vc = curr_vc,
                            ec = curr_ec,
                            fam_name = fam_name,
                            subfam_name = gen_name,
                            ind = i,
                            ext = self.gen_ext
                        )

                        curr_gen_files += [curr_onet]
                        
                        self.gen_args_list += [
                            {
                                "prg": self.gen_dict[fam_name][gen_name]['generator_path'],
                                "nodes":curr_vc,
                                "edges":curr_ec,
                                "onet":curr_onet,
                                "undir":self.undir,
                                
                                "family":fam_name,
                                "generator":gen_name,
                            }
                        ]

                    curr_ind = i

                    #if "onets" in self.gen_dict[fam_name][gen_name].keys():
                    #    self.gen_dict[fam_name][gen_name]["onets"] += curr_gen_files
                    #else:
                    #    self.gen_dict[fam_name][gen_name]["onets"] = curr_gen_files
                    
                    curr_files += curr_gen_files
                    
                files_paths += curr_files
                
        return files_paths
        
    def _init_files(self
                   , files_paths
                  ):
        
        # create each directories
        directory_n_m_fs = np.unique(["/".join(f.split("/")[:-1])for f in files_paths])
        for d in directory_n_m_fs:
            make_dir(d)
            
        # create each file
        #touch_files_str = "touch "+";\ntouch ".join(files_paths)
        #subprocess.run(touch_files_str, shell=True)
        
        return True
    
    def _generate_nets(self
                       , files_paths:list = None
                      ):
        if files_paths is None:
            gen_args_list = self.gen_args_list
        else:
            gen_args_list = [
                args for args in self.gen_args_list
                if args["onet"] in files_paths
            ]
            
        pool = mp.Pool(processes=self.n_processes)
        times = pool.map(gen_from_args, gen_args_list)

        # clean up
        pool.close()
        pool.join()
        
        return times
    
    def make_embedder(self,
                      **kwargs
                     ):
        #raise NotImplementedError("TODO: implement SynNets inheritance.")
        return SynNetsEmbedder(
            gen_args_list = self.gen_args_list,
            **kwargs
        )
    
    

class SynNetsEmbedder(object):
    def __init__(
        self,
        gen_args_list,
        embedding_method=None, # embedd nets
        reducer=None, # map to 2 dimensions --> for visualisation
        to_analyzer:bool = False,
    ):
        # 0. check all required metadata
        required_keys = ["onet"]
        availabled_metadata = check_meta_args(gen_args_list, required_keys)
        assert availabled_metadata, "Missing required arguments from: {}".format(required_keys)
        # ##############################
        self.generated_args = gen_args_list
        
        self.embedder = embedding_method
        
        if not reducer is None:
            self.reduce = True
            self.reducer = reducer
        else:
            self.reduce = False
            
            
        # compute embeddings
        self.embed_from_args()
        
        if to_analyzer:
            return self.make_analyzer()
        
        
    def embed_from_args(self
                       ):
        
        embs = np.array([
            self.embedder(load_net(arg["onet"], directed=False)) 
            for arg in self.generated_args
        ]
        )
        
        if self.reduce:
            embs = self.reducer.fit_transform(embs)
            
        for i, arg in enumerate(self.generated_args):
            arg["emb"] = embs[i]
        
        return
    
    def make_analyzer(self,
                      **kwargs
                     ):
        #raise NotImplementedError("TODO: implement SynNets inheritance.")
        return SynEmbAnalyzer(
            emb_args_list = self.generated_args,
            **kwargs
        )

class SynEmbAnalyzer():
    def __init__(
        self,
        emb_args_list,
        do_plot = False,
    ):
        # 0. check all required metadata
        required_keys = required_keys = ["nodes", "edges", "emb", "family", "generator", "onet"]
        
        availabled_metadata = check_meta_args(emb_args_list, required_keys)
        assert availabled_metadata, "Missing required arguments from: {}".format(required_keys)
        
        required_dim = 2
        assert np.all([len(a["emb"]==2) for a in emb_args_list]), "Embeddings MUST be 2-dimensional for the analysis."
        # ##############################
        
        # 1.1 retrieve (v,e) pairs & make grid:  ve -> true/false
        computed_ve_pairs = list(set([(a["nodes"], a["edges"]) for a in emb_args_list]))
        computed_ve_pairs = sorted(computed_ve_pairs, key=lambda element: (element[0], element[1]))
        self.ve_pairs = computed_ve_pairs
        
        self.unique_vs = sorted(np.unique([ve[0] for ve in computed_ve_pairs]))
        self.unique_es = sorted(np.unique([ve[1] for ve in computed_ve_pairs]))
        
        self.computed_grid = np.array(
            [
                [
                    (v,e) if (v,e) in self.ve_pairs else False
                    for e in self.unique_es
                ]
                for v in self.unique_vs
            ]
        , dtype=object)
        
        # 1.2 organize data
        self.ve_dicts = {}
        for ve in self.ve_pairs:
            
            embs, fams, onets = get_ve_values_form_gen_args(
                args_list = emb_args_list,
                ve_pair = ve,
                keys = ["emb", "family", "onet"]
            )
            
            self.ve_dicts[ve] = {
                "ve": ve,
                "embs": embs,
                "labels": fams,
                "srcs": onets
            }
            
        # 2.1
        self.computed_scores = False
        
            
    def visualise(self,
                  visualise_ve=visualise_kde,
                  single_plot_size=4,
                  #figsize=(20,20),
                  return_fig = False,
                  dark_mode = False,
                 ):
        # init figure --> make grid
        nrows, ncols = self.computed_grid.shape[0], self.computed_grid.shape[1]
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            
            figsize=(single_plot_size*ncols, single_plot_size*nrows)
        )
        
        #if len(self.computed_grid)*len(self.computed_grid[0]) > 1:
        for i in range(nrows):
            for j in range(ncols):
                ve = self.computed_grid[i,j]
                try:
                    ve = tuple(ve)
                except:
                    pass
                # get current ax to plot one:
                curr_ax = None
                if ((nrows>1) and (ncols>1)):
                    curr_ax = axes[i,j]  
                elif ((nrows==1) and (ncols==1)):
                    curr_ax = axes
                elif ncols==1:
                    curr_ax = axes[i]
                else:
                    curr_ax = axes[j]
                    
                if ve:
                    self.single_viz(
                        ve=ve,
                        visualisation_method=visualise_ve,
                        ax=curr_ax,#axes[i,j],
                        title=ve,
                        dark_mode = dark_mode,
                    )
                else:
                    remove_frame(ax=curr_ax)
                        
        """else:
            ve = tuple(self.computed_grid[0,0])
            self.single_viz(
                ve=ve,
                visualisation_method=visualise_ve,
                ax=axes,
                title=ve,
                dark_mode = dark_mode,
            )
        """
                    
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            return
    
    def single_viz(
        self,
        ve,
        visualisation_method, # must have <ax>, <ve_dict> and <title> as args
        ax = None,
        title = None,
        dark_mode = False,
        reducer=None,
    ):
        ve_dict = self.ve_dicts[ve]
        if ax is None:
            fig, ax = plt.subplots()

        visualisation_method(
            ve_dict=ve_dict,
            ax = ax,
            title = title,
            reducer=reducer,
        )

        remove_frame(ax)

        if title is None:
            title = ve

        ax.set_title(title
                     , loc='left'
                     , fontweight='semibold'
                     , color="w" if dark_mode else "k"
                    )

        return ax
    
    def get_families(self):
        ve_key = list(self.ve_dicts.keys())[0]
        families = np.unique(self.ve_dicts[ve_key]["labels"])
        return families
    
    def get_vertices(self):
        return np.unique([k[0] for k in self.ve_dicts.keys()])
    
    def get_edges(self):
        return np.unique([k[1] for k in self.ve_dicts.keys()])
    
    def get_sparsities(self, sparsity_round:int=3):
        return np.unique([np.round(sparsity(k[0], k[1]), sparsity_round) for k in self.ve_dicts.keys()])
    
    def compute_scores(
        self,
        families:list=[],
        metric_name:str="silhouette",
    ):
        if len(families)<2:
            families = self.get_families()
            
        pairwise_metric, overall_metric = str_to_metrics(metric_name)
        
        for k, v in self.ve_dicts.items():
            # compute overall score:
            v["overall_score"] = overall_metric(v["embs"], v["labels"])

            # compute pairwise scores:
            pairwise_dict = {fam: None for fam in families}
            for curr_fam in families:
                curr_fam_dict = {}
                curr_fam_embs = np.array([e for e, f in zip(v["embs"], v["labels"]) 
                                          if f == curr_fam
                                         ]
                                        )
                for other_fam in families:
                    if other_fam == curr_fam:
                        pass
                    else:
                        other_fam_embs = np.array([e for e, f in zip(v["embs"], v["labels"]) 
                                                   if f == other_fam
                                                  ]
                                                 )
                        curr_fam_dict[other_fam] = pairwise_metric(curr_fam_embs, other_fam_embs)

                pairwise_dict[curr_fam] = curr_fam_dict

            v["pairwise_scores"] = pairwise_dict
            
        self.computed_scores = True
            
        return
    
    def classify(
        self,
    ):
        for k, v in self.ve_dicts.items():
            # compute overall score:
            
            res = svc_classify(
                np.array(v["embs"])
                , np.array(v["labels"])
            )
            
            for res_k in res.keys():
                v[res_k] = res[res_k]
            
        return 
    
    def get_overall_scores(
        self,
        common_p:bool=False,
        sparsity_round:int=3,
        get_score=None,
    ):
        if not self.computed_scores:
            self.compute_scores
        # # # # # # #
        
        dict_scores = {}
        for k, v in self.ve_dicts.items():
            r = k[0]
            c = k[1]
            if common_p:
                c = np.round(sparsity(r,c), sparsity_round)
            
            if not r in dict_scores.keys():
                dict_scores[r] = {}

            if get_score is None:
                score = v["overall_score"]
            elif isinstance(get_score, str):
                score = v[get_score]
            else:
                score = get_score(v)
            dict_scores[r][c] = score
        
        overall_scores = pd.DataFrame(dict_scores).T
        overall_scores = overall_scores.reindex(sorted(overall_scores.columns, reverse=False), axis=1)
        overall_scores = overall_scores.reindex(sorted(overall_scores.index, reverse=True), axis=0)
        
        return overall_scores
    
    def plot_overall_scores(
        self,
        overall_scores:pd.DataFrame=None,
        common_p:bool=True,
        sparsity_round:int=3,
        cmap = "GnBu",
        ax = None
    ):
        if overall_scores is None:
            overall_scores = self.get_overall_scores(common_p, sparsity_round)
        
        sns.heatmap(
            overall_scores.values,#[::-1],
            vmin=0, vmax=1,
            cmap="GnBu",
            yticklabels=overall_scores.index,
            xticklabels=overall_scores.columns,
            ax = ax
        )
        
        return
    
    def plot_families_scores(
        self,
        families:list=[],
        fam_to_color_dict:dict={},
        plot_sparsities:bool=True,
        sparsity_round:int=3,
    ):
        if len(families)<2:
            families = self.get_families()
        if not self.computed_scores:
            self.compute_scores(families = families)
            
        use_fam_colors = np.all([f in fam_to_color_dict.keys() for f in families])
        
        nodes = self.get_vertices()
        if plot_sparsities:
            sparsities = self.get_sparsities()
        else:
            x_plot = self.get_edges()
        nb_fams = len(families)
        nb_nodes_gen = len(nodes)

        fig, ax = plt.subplots(figsize= (6*nb_fams, 4*nb_nodes_gen)
                               , nrows=nb_nodes_gen
                               , ncols=nb_fams
                               , sharey=True
                               , sharex=plot_sparsities
                              )

        for j, fam in enumerate(families):
            fam_dict = {ve: None for ve in self.ve_dicts.keys()}
            for k, v in self.ve_dicts.items():
                fam_dict[k] = v["pairwise_scores"][fam]

            other_fams = list(fam_dict[list(fam_dict.keys())[0]].keys())

            for k, v in self.ve_dicts.items():
                fam_dict[k] = v["pairwise_scores"][fam]

            for i, nc in enumerate(nodes):

                fam_dict_nc = {k:v for k, v in fam_dict.items() if k[0]==nc}
                n_edges = [k[1] for k in fam_dict_nc.keys()]
                sparsities = [np.round(sparsity(nc, ne), sparsity_round) for ne in n_edges]#edges_to_sparsity(nc, n_edges)

                for f in other_fams:

                    ax[i,j].plot(
                        sparsities if plot_sparsities else n_edges,
                        [fam_dict_nc[(nc, ne)][f] for ne in n_edges],
                        marker = 'o',
                        label = f,
                        c=fam_to_color_dict[f] if use_fam_colors else None,
                    )

                ax[i,j].set_title("{} ({})".format(fam, nc)
                                  , fontweight="semibold"
                                  , c=fam_to_color_dict[fam] if use_fam_colors else 'k',
                         )
        return
                
    def plot_pairwise_scores(
        self,
        families:list=[],
        common_p:bool=True,
        sparsity_round:int=3,
    ):
        def get_fam1_fam2_scores(v, fam1, fam2):
            return v["pairwise_scores"][fam1][fam2]
        
        if len(families)<2:
            families = self.get_families()
        if not self.computed_scores:
            self.compute_scores(families = families)
            
        nb_fams = len(families)
        
        
        fig, ax = plt.subplots(figsize=(5*nb_fams,5*nb_fams)
                               , nrows=nb_fams, ncols=nb_fams
                              )

        for i, fam in enumerate(families):
            for j, other_fam in enumerate(families):
                if other_fam == fam:
                    pass
                else:
                    
                    fam_scores = self.get_overall_scores(
                        common_p=common_p,
                        sparsity_round=sparsity_round,
                        get_score=lambda v: get_fam1_fam2_scores(v, fam1=fam, fam2=other_fam),
                    )

                    self.plot_overall_scores(
                        overall_scores = fam_scores,
                        ax = ax[i,j],
                    )

                    ax[i,j].set_title('{}-{}'.format(fam, other_fam))

        
        return
        
        
        