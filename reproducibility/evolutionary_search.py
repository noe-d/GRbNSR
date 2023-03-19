"""
                    ___   ____ _____ __
  ___ _   ______   |__ \ / __ <  / // /
 / _ \ | / / __ \  __/ // / / / / // /_
/  __/ |/ / /_/ / / __// /_/ / /__  __/
\___/|___/\____(_)____/\____/_/  /_/   
                                       
"""
import os
import sys

import numpy as np
import networkx as nx
from tqdm.auto import tqdm

import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# to allow imports
sys.path.append("./")

from reproducibility.utils import *

from synthetic.net import load_net#, load_from_nx, comprehensive_load_net
from synthetic.generator import load_generator

from synthetic.distances import DistancesToNet, Norm
from synthetic.commands.evo import Evolve

from synthetic.deep_distance import *

from synthetic.consts import (DEFAULT_GENERATIONS, DEFAULT_SAMPLE_RATE,
                              DEFAULT_BINS, DEFAULT_MAX_DIST,
                              DEFAULT_TOLERANCE, DEFAULT_GEN_TYPE,
                              DEFAULT_NORM_SAMPLES, 
                              DEFAULT_NODES, DEFAULT_EDGES, DEFAULT_RUNS
                             )

from synthetic.commands.command import get_stat_dist_types, arg_with_default

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Storage
## in
_DATA_FOLDER = "space_seg/data/data_v2/v100_e552/ER/" # obs | prog | synth
## out
_OUTPUT_PATH = "data/repro_trad_degs/ER/"

# experiment args
_DEFAULT_ARGS = {
    "data_path":_DATA_FOLDER,
    "inets": [
        "c1_1.pickle",
        "c1_2.pickle",
        "c1_3.pickle",
    ],
    "output_path":_OUTPUT_PATH,
    
    "directed":False, 
    
    "n_rep":10, # number of experiments run
    
    "do_parallel":False,
    "n_processes":1, # if do_parallel == True: number of parallel processes
    
    "model_path":"./DL_module/saved/best_models/trad_degs/",
    "dist_type":"euclidean",
}


# Visualisation
_DEFAULT_NX_VIZ = {
    "with_labels":False,
    "node_shape":"o",
    "node_size":50,
    "alpha":0.9,
    "linewidths":0.1,
    "edge_color":"grey",
    "width":0.5,
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_indiviual_exp(inet, output_folder, directed=False):
    print("TODO: individual exp", output_folder)
    return

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
#    ________  .....
#   / ___/ _ \/ ...;
#  / /  /  __(... ) 
# /_/   \___/....(_)
#                  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
""" DEPRECATED """

""" 
def best_old_score(exp:str,
                   args:dict={},
                   exps_dir:str = _DATA_FOLDER,
                  ):
    # load obs
    old_net = load_net(args["inet"], directed = not args["undir"])
    
    # instantiate distance_to_net
    dist_to_net = DistancesToNet(
        net = old_net,
        stat_dist_types = get_stat_dist_types(args),
        bins = arg_with_default(args, 'bins', DEFAULT_BINS),
        max_dist = arg_with_default(args, 'maxdist', DEFAULT_MAX_DIST),
        norm=Norm.ER_MEAN_RATIO,
        norm_samples=DEFAULT_NORM_SAMPLES,
        rw=True,
    )
    
    # load old best net
    old_exp_path = exps_dir+"synth/{net}-synth.edges".format(net=exp)
    old_net = load_net(old_exp_path, directed = not args["undir"])
    
    # compute distance ! 
    old_loss = dist_to_net.compute(old_net)
    
    return old_loss
"""

"""
def pairwise_gen_dist(args):
    # from commands/eval_distances
    
    prog1 = args['prg']
    prog2 = args['prg2']

    sr = arg_with_default(args, 'sr', DEFAULT_SAMPLE_RATE)
    directed = not args['undir']
    nodes = arg_with_default(args, 'nodes', DEFAULT_NODES)
    edges = arg_with_default(args, 'edges', DEFAULT_EDGES)
    gentype = arg_with_default(args, 'gentype', DEFAULT_GEN_TYPE)

    gen1 = load_generator(prog1, directed, gentype)
    gen2 = load_generator(prog2, directed, gentype)

    gen1.run(
        nodes, 
        edges, 
        sr, 
        shadow=gen2,
    )
    dist1 = gen1.eval_distance
    gen2.run(
        nodes, 
        edges, 
        sr, 
        shadow=gen1,
    )
    dist2 = gen2.eval_distance
    dist = (dist1 + dist2) / 2
    
    return dist
    
"""

"""
def generators_distances_to_ref(net,
                                ref_prog:str,
                                list_of_progs:list,
                               ):
    args_dist = {
        "prg":ref_prog,
        "prg2":"",
        "undir": True,
        "nodes":net.graph.vcount(),
        "edges":net.graph.ecount()
    }
    
    distances = []
    for p in list_of_progs:
        args_dist["prg2"] = p
        
        distances += [pairwise_gen_dist(args_dist)]
    
    return distances
"""
    
"""
def generators_distances(net,
                         list_of_progs:list
                        ):
    
    nb_progs = len(list_of_progs)
    dist_mat = np.zeros((nb_progs, nb_progs))
    
    # custom loading bar
    with tqdm(desc="Pairwise distances", total=int((nb_progs*(nb_progs-1))/2)) as pbar:
        for i in range(nb_progs):
            dists_to_i = generators_distances_to_ref(net=net,
                                                     ref_prog=list_of_progs[i],
                                                     list_of_progs=list_of_progs[:i],
                                                    )

            dist_mat[i] = dists_to_i + [0]*(nb_progs-i)
            
            pbar.update(i)
 
    #symetrize
    dist_mat = (dist_mat+dist_mat.T)/2.
            
    return dist_mat
"""

"""
def generator_fitness(args
                      , fit_optim = np.max
                      , dist2net = None
                     ):
    ### commands/fit ###
    
    netfile = args['inet']
    
    sr = arg_with_default(args, 'sr', DEFAULT_SAMPLE_RATE)
    bins = arg_with_default(args, 'bins', DEFAULT_BINS)
    max_dist = arg_with_default(args, 'maxdist', DEFAULT_MAX_DIST)
    directed = not args['undir']
    
    prog = args['prg']
    runs = arg_with_default(args, 'runs', DEFAULT_RUNS)
    gen_type = arg_with_default(args, 'gentype', DEFAULT_GEN_TYPE)

    # load net
    net = load_net(netfile, directed)
    
    # create fitness calculator
    # TODO: norm samples configurable
    if dist2net is None:
        fitness = DistancesToNet(
            net=net,                      
            stat_dist_types=get_stat_dist_types(args),#[:3],
            bins=bins,                     
            max_dist=max_dist,                      
            norm=Norm.ER_MEAN_RATIO,                     
            norm_samples=30,#DEFAULT_NORM_SAMPLES,
            rw=True,
        )
    else:
        fitness = dist2net
    
    fit_values = []
    gen = load_generator(prog, directed, gen_type)
    vcount = net.graph.vcount()
    ecount = net.graph.ecount()
    mean_dist = []
    for i in range(runs):

        synth_net = gen.run(
            vcount, 
            ecount, 
            sr,
            #do_connect=True,
        )
        distances = fitness.compute(synth_net)
        
        fit_values += [fit_optim(distances)]
        
        mean_dist += [distances]
        
        
    mean_dist = np.mean(mean_dist, axis=0)
    for s,v,f in zip(fitness.stat_dist_types,fitness.norm_values,mean_dist):
        print(s,v,f)
                
            
            
    return fit_values #np.mean(fit_values), np.std(fit_values)
    
"""

"""
def generators_fitness(args
                       , list_of_progs:list
                       , fit_otpim=np.max
                      ):
    
    netfile = args['inet']
    
    bins = arg_with_default(args, 'bins', DEFAULT_BINS)
    max_dist = arg_with_default(args, 'maxdist', DEFAULT_MAX_DIST)
    directed = not args["undir"]
    
    # load net
    net = load_net(netfile, directed)
    
    # create fitness calculator
    # for more fair comparisons (+ faster computation):
    #       compute normalisation w/ regards to the same values
    dist2net = DistancesToNet(
        net=net,                      
        stat_dist_types=get_stat_dist_types(args),#[:3],
        bins=bins,                     
        max_dist=max_dist,                      
        norm=Norm.ER_MEAN_RATIO,                     
        norm_samples=DEFAULT_NORM_SAMPLES,
        rw=True,
    )
    
    whole_fits = []
    for p in tqdm(list_of_progs):
        args["prg"] = p
        
        fits = generator_fitness(args, fit_otpim, dist2net)
        whole_fits += [fits]
        
    return whole_fits

def get_fitness_dissim(list_of_progs):
    # get fitness --> retrieve best
    # compute dissims w/ regards to best
    return
    
"""
            
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
#         _    
#  _   __(_)___
# | | / / /_  /
# | |/ / / / /_
# |___/_/ /___/
# 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%

def nx_load_obs_new_old_nets(exp:str
                             , best_run_id:int=None
                             , exps_dir = _DATA_FOLDER
                             , res_dir = _OUTPUT_PATH
                             , directed_net = False
                            ):
    if best_run_id is None:
        best_run = retrieve_best_runs( 
            results_dir=args["output_path"], 
            n_exps=args["n_rep"],
            experiments_names=[exp],

            csv_loss_name="evo.csv",

            loss_col = "best_fit",           
            optim="min",                    
        )
        best_run_id = best_run[exp]["best_run_id"]
        
    best_new_graph = res_dir+"/{}.{}/bestnet.gml".format(exp, best_run_id)
    best_old_graph = exps_dir+"/synth/{}-synth.edges".format(exp)

    obs_graph = exps_dir+"/obs/{}.gml".format(exp)
    
    networkx_graphs = [load_net(gp, directed=directed_net).graph.to_networkx()
                       for gp in [obs_graph, best_new_graph, best_old_graph]
                      ]
    return networkx_graphs

def triple_viz(exp,
               networkxs, 
               fig_kwargs={"figsize":(10,10)
                          },
               nx_kwargs=_DEFAULT_NX_VIZ,
               colors=["#4953AB"
                       ,"#F4B886"
                       ,"#6C8C69"
                      ]
              ):
    
    fig = plt.figure(constrained_layout=True, **fig_kwargs)

    #gs = GridSpec(3, 2, figure=fig)
    #ax_obs = fig.add_subplot(gs[:2, :])
    #ax_new = fig.add_subplot(gs[2, 0])
    #ax_old = fig.add_subplot(gs[2, 1])
    
    gs = GridSpec(3, 3, figure=fig)
    ax_obs = fig.add_subplot(gs[1])
    ax_new = fig.add_subplot(gs[2])
    ax_old = fig.add_subplot(gs[0])


    for g, ax, t, c in zip(networkxs
                           , [ax_obs, ax_new, ax_old]
                           , ["obs","new","old"]
                           , colors
                          ):
        nx.draw(g, ax=ax
                , node_color=c
                , **nx_kwargs
               )
        ax.set_title(t)

    fig.suptitle(exp, fontweight="bold")
    plt.show()
    
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
import multiprocessing as mp

from synthetic.consts import (DEFAULT_GENERATIONS, DEFAULT_SAMPLE_RATE, DEFAULT_BINS, DEFAULT_MAX_DIST,
                              DEFAULT_TOLERANCE, DEFAULT_GEN_TYPE, DEFAULT_NORM_SAMPLES)
from synthetic.net import load_net
from synthetic.generator import create_generator
from synthetic.distances import DistancesToNet, Norm
from synthetic.evo import Evo
from synthetic.commands.command import Command, arg_with_default, get_stat_dist_types
from synthetic.utils import set_seed

class MutliEvolve(Command):
    def __init__(self, cli_name):
        Command.__init__(self, cli_name)
        self.name = 'multi_evo'
        self.description = 'evolve network generator'
        self.mandatory_args = ['inet', 'odir', 'n_reps']
        self.optional_args = ['undir', 'gens', 'sr', 'bins', 'maxdist', 'tolerance', 'gentype', 'rw']
        
    def run(self
            , args
            , n_reps:int=16
            , n_processes:int=16
            , og_seed:int=None
            , do_parallel:bool=True
            , old_netfile:str=None
            , save_all:int=0
            , verbosity:int=0
           ):
        self.error_msg = None
        
        self.do_parallel = do_parallel
        self.save_all = save_all
        self.verbosity = verbosity
        
        deep_dist = False
        if "model_path" in args.keys():
            if not args["model_path"] is None:
                deep_dist = True

        # net info
        netfile = args['inet']
        outdir = args['odir']
        directed = not args['undir']
        
        # load net
        self.net = load_net(netfile, directed)
        
        # generation infos
        tolerance = arg_with_default(args, 'tolerance', DEFAULT_TOLERANCE)
        generations = arg_with_default(args, 'gens', DEFAULT_GENERATIONS)
        sr = arg_with_default(args, 'sr', DEFAULT_SAMPLE_RATE)
        gen_type = arg_with_default(args, 'gentype', DEFAULT_GEN_TYPE)
        
        # create base generator
        self.base_generator = create_generator(directed, gen_type)
        if self.base_generator is None:
            self.error_msg = 'unknown generator type: {}'.format(gen_type)
            return False
        
        # set the seed if given
        self.og_seed = og_seed
        
        # to create evolutionary search
        self.tolerance = tolerance
        self.sr = sr
        self.base_odir = outdir
        self.generations = generations
        
        # some reports to screen
        info_params = [
            'target net: {}'.format(netfile),
            'stable generations: {}'.format(generations),
            'directed: {}'.format(directed),
            'target net node count: {}'.format(self.net.graph.vcount()),
            'target net edge count: {}'.format(self.net.graph.ecount()),
            'tolerance: {}'.format(tolerance),
            'deep distance : {}'.format(deep_dist),
            'OG seed: {}'.format(og_seed)
        ]
        self.info_str = '\n'.join(info_params)
        
        if _OG_SEED is not None:
            set_seed(_OG_SEED)
        
        # distance:
        ## trad. (2014-like)
        print('Initialising distance metric...')
        
        if not deep_dist:
            bins = arg_with_default(args, 'bins', DEFAULT_BINS)
            max_dist = arg_with_default(args, 'maxdist', DEFAULT_MAX_DIST)

            rw = args['rw']
        
            self.dist2net = DistancesToNet(
                self.net, 
                get_stat_dist_types(args), 
                bins, 
                max_dist, 
                rw, 
                norm=Norm.ER_MEAN_RATIO,
                norm_samples=DEFAULT_NORM_SAMPLES,
            )
            
        else:
            model_path = args["model_path"]
            distance_type = args["dist_type"]
            
            self.dist2net = DeepDistancesToNet(   
                net = self.net,
                model = model_path,
                dist_type = distance_type,
                norm = Norm.NONE,
            )
            
        
        #if not old_netfile is None:
        #    old_net = load_net(old_netfile, directed)
        #    old_fitness = self.compute_net_fitness(old_net)
        #    print("Best old fitness : {}".format(old_fitness))
            
        inds = np.arange(0, n_reps)
        
        if self.do_parallel:
            pool = mp.Pool(processes=n_processes)
            times = pool.map(self.single_run, inds)

            # clean up
            pool.close()
            pool.join()
            
        else:
            for i in inds:
                self.single_run(ind=i)
            

        return True
    
    def compute_net_fitness(self,
                            net,
                            fit_method=None, #def in evo --> max
                           ):
        
        fitness = self.dist2net.compute(net)
        
        if fit_method is None:
            return fitness
        else: 
            return fit_method(fitness)
    
    def single_run(self,
                   ind:int,
                  ):
        
        outdir = self.base_odir+'.{}'.format(ind)
        make_folder(outdir)
        
        # write experiment params to file
        with open('{}/params.txt'.format(outdir), 'w') as text_file:
            text_file.write(self.info_str)
            
            
        if self.og_seed is not None:
            set_seed(self.og_seed+ind)
            
        evo = Evo(
            self.net, 
            self.dist2net, 
            self.generations, 
            self.tolerance, 
            self.base_generator, 
            outdir, 
            self.sr,
            verbosity=self.verbosity,
            save_all=self.save_all,
        )
        
        # run search
        evo.run()
        
        return True
    
    
    
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   _       
#   _ __ ___   __ _(_)_ __  
#  | '_ ` _ \ / _` | | '_ \ 
#  | | | | | | (_| | | | | |
#  |_| |_| |_|\__,_|_|_| |_|
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%

_UNDIR = True
_OG_SEED = 0

def main(args):
    # 0. preparation
    make_folder(args["output_path"])
    
    #nets_files = get_list_of_nets(folder=args["data_path"], src_subfolder=args["data_folder"])
    #nets_files = ["words.gml"]#["words.gml"]#["power.gml"]#
    
    nets_files = args["inets"]
    
    #nets_paths = [args["data_path"]+args["data_folder"]+"/"+f for f in nets_files]
    nets_paths = [args["data_path"]+f for f in nets_files]
    print("Nets files : ", nets_files)
    
    # retrieve &/| build experiment args from args
    output_exp = args["output_path"]+"{exp_name}.{exp_num}"
    if "model_path" in args:
        deep_dist = True
        model_path = args["model_path"]
        dist_type = args["dist_type"]
    
    experiments = []
    
    original_seed = args["seed"] if "seed" in args.keys() else _OG_SEED
    
    # 1. run all experiments
    for flnm, path in zip(nets_files, nets_paths): # for each network of the experiment

        # load the network
        #net = comprehensive_load_net(path, args["directed"])
        exp = flnm.split(".")[0]
        experiments += [exp]


        output_exp = args["output_path"]+"{exp_name}".format(exp_name=exp)

        exp_args = {
            "inet":path,
            "verbosity": args["verbosity"],
            "save_all": args["save_all"],
            "odir":output_exp,
            "undir": not args["directed"],
            "gens":DEFAULT_GENERATIONS,
            "sr":DEFAULT_SAMPLE_RATE,
            "tolerance":DEFAULT_TOLERANCE,
            "rw":args["rw"],
        }
        
        if not deep_dist:
            evo_args = {
                "bins":DEFAULT_BINS,
                "maxdist":DEFAULT_MAX_DIST,
                "rw":True,
            }
        else:
            evo_args = {
                "model_path":model_path,
                "dist_type":dist_type,
            }
            
        exp_args.update(evo_args)

        multievo = MutliEvolve("multievo")

        # 📟 compute old best
        #oldfile = args["data_path"]+"synth/{net}-synth.edges".format(net=exp)

        # 🏃 run experiments
        multievo.run(
            args = exp_args,
            n_reps = args["n_rep"],
            n_processes = args["n_processes"],
            og_seed = original_seed,#_OG_SEED, 
            do_parallel = args["do_parallel"],
            save_all = args["save_all"],
            verbosity = args["verbosity"],
            #old_netfile = oldfile,
        )

            
    # retrieve best NEW runs
    best_runs = retrieve_best_runs( # get dict {exp: {best_run_id: X, best_run_path: X, best_fit: X, n_runs: X}}
        results_dir=args["output_path"], #  FO-
        n_exps=args["n_rep"],            # -LD-
        experiments_names=experiments,   # -ER
        
        csv_loss_name="evo.csv",         # FILE
        
        loss_col = "best_fit",           # LO-
        optim="min",                     # -SS
    )
    
    print(best_runs)
    
    
    # analyse results (mixing metrics)
            
    return

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
import argparse

if __name__ == "__main__":
    # 0. parse agrs and retrieve list of networks
    args = _DEFAULT_ARGS
    
    parser = argparse.ArgumentParser()
    
    """
    parser.add_argument(
        '--verbose'
        , action='store_true'
        , help=''
        , default=_DEFAULT_ARGS["verbose"]
    )
    """
    
    # output_path
    parser.add_argument(
        '--output_path'
        , help=''
        , type=str
        , default=_DEFAULT_ARGS["output_path"] if "output_path" in _DEFAULT_ARGS.keys() else None
    )
    # verbosity
    parser.add_argument(
        '--verbosity'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["verbosity"] if "verbosity" in _DEFAULT_ARGS.keys() else 0
    )
    # verbosity
    parser.add_argument(
        '--save_all'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["save_all"] if "save_all" in _DEFAULT_ARGS.keys() else 0
    )
    # data_path
    parser.add_argument(
        '--data_path'
        , help=''
        , type=str
        , default=_DEFAULT_ARGS["data_path"] if "data_path" in _DEFAULT_ARGS.keys() else None
    )
    # inets
    parser.add_argument(
        '--inets'
        , help=''
        , nargs='+'
        , default=_DEFAULT_ARGS["inets"] if "inets" in _DEFAULT_ARGS.keys() else None
    )
    # directed
    parser.add_argument(
        '--directed'
        , help=''
        , action='store_true'
        , default=_DEFAULT_ARGS["directed"] if "directed" in _DEFAULT_ARGS.keys() else False
    )
    # n_rep
    parser.add_argument(
        '--n_rep'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["n_rep"] if "n_rep" in _DEFAULT_ARGS.keys() else None
    )
    # seed
    parser.add_argument(
        '--seed'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["seed"] if "seed" in _DEFAULT_ARGS.keys() else None
    )
    # do_parallel
    parser.add_argument(
        '--do_parallel'
        , help=''
        , action='store_true'
        , default=_DEFAULT_ARGS["do_parallel"] if "do_parallel" in _DEFAULT_ARGS.keys() else False
    )
    # n_processes
    parser.add_argument(
        '--n_processes'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["n_processes"] if "n_processes" in _DEFAULT_ARGS.keys() else None
    )
    # tolerance
    parser.add_argument(
        '--tolerance'
        , help=''
        , type=float
        , default=_DEFAULT_ARGS["tolerance"] if "tolerance" in _DEFAULT_ARGS.keys() else None
    )
    # gens
    parser.add_argument(
        '--gens'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["gens"] if "gens" in _DEFAULT_ARGS.keys() else None
    )
    # sr
    parser.add_argument(
        '--sr'
        , help=''
        , type=float
        , default=_DEFAULT_ARGS["sr"] if "sr" in _DEFAULT_ARGS.keys() else None
    )
    # gen_type
    parser.add_argument(
        '--gen_type'
        , help=''
        , type=str
        , default=_DEFAULT_ARGS["gen_type"] if "gen_type" in _DEFAULT_ARGS.keys() else None
    )
    # model_path
    parser.add_argument(
        '--model_path'
        , help=''
        , type=str
        , default=_DEFAULT_ARGS["model_path"] if "model_path" in _DEFAULT_ARGS.keys() else None
    )
    # dist_type
    parser.add_argument(
        '--dist_type'
        , help=''
        , type=str
        , default=_DEFAULT_ARGS["dist_type"] if "dist_type" in _DEFAULT_ARGS.keys() else None
    )
    # norm_samples
    parser.add_argument(
        '--norm_samples'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["norm_samples"] if "norm_samples" in _DEFAULT_ARGS.keys() else None
    )
    # bins
    parser.add_argument(
        '--bins'
        , help=''
        , type=int
        , default=_DEFAULT_ARGS["bins"] if "bins" in _DEFAULT_ARGS.keys() else None
    )
    # max_dist
    parser.add_argument(
        '--max_dist'
        , help=''
        , type=float
        , default=_DEFAULT_ARGS["max_dist"] if "max_dist" in _DEFAULT_ARGS.keys() else None
    )
    # rw
    parser.add_argument(
        '--rw'
        , help=''
        , action='store_true' 
        , default=_DEFAULT_ARGS["rw"] if "rw" in _DEFAULT_ARGS.keys() else None
    )
    
    # Force trad
    parser.add_argument(
        '--force_trad'
        , help=''
        , action='store_true'
        , default=False
    )

    args = vars(parser.parse_args())
    
    # --> force
    if args["force_trad"]:
        args["model_path"] = None
    
    # 1. loop for all network
    
    # 2. meta results quality study
    
    #print(args)
    
    main(args)