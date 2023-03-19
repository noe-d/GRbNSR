"""
,---.  ,---..-./`)    .-'''-.   ___    _    ____      .---.      
|   /  |   |\ .-.')  / _     \.'   |  | | .'  __ `.   | ,_|      
|  |   |  .'/ `-' \ (`' )/`--'|   .'  | |/   '  \  \,-./  )      
|  | _ |  |  `-'`"`(_ o _).   .'  '_  | ||___|  /  |\  '_ '`)    
|  _( )_  |  .---.  (_,_). '. '   ( \.-.|   _.-`   | > (_)  )    
\ (_ o._) /  |   | .---.  \  :' (`. _` /|.'   _    |(  .  .-'    
 \ (_,_) /   |   | \    `-'  || (_ (_) _)|  _( )_  | `-'`-'|___  
  \     /    |   |  \       /  \ /  . \ /\ (_ o _) /  |        \ 
   `---`     '---'   `-...-'    ``-'`-''  '.(_,_).'   `--------` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~> class <GRVisualiser>
   _                                                              
 _( )_
(_ o _) 2D-projection of Graph Representation.
 (_,_)
   _                                                             
 _( )_
(_ o _) Explorative plot to visualise the Representation space,
 (_,_)                   and individual instances inhabiting it.
 
"""


from .viz_util import *

class GRVisualiser():
    # =================
    # INIT & ATTRIBUTES
    # =================
    def __init__(
        self,
        
        graphs:list,
        embeddings:list,
        labels:list,
        og_format:str,
        # dataset, model
        
        reducer_type:str="manifold.TSNE",
        reducer_args:dict={},
        
        make_interactive:bool=False,
        dataset_name:str=None,
        model_name:str=None,
        
        seed:int = 0
    ):
        self._graphs = graphs
        self._embeddings = embeddings
        self._labels = labels
        self._og_format = og_format
        
        if not reducer_type is None:
            self._reducer, self._reducer_type = get_reducer_from_sklearn(reducer_type)
            self._reducer = self._reducer(2, **reducer_args)
            # project embeddings:
            self.project_embs()
        else:
            self._reducer, self._reducer_type = None, None
            self._projected_embeddings = self._embeddings
        
        self._make_df()
        
        self._dataset_name = dataset_name
        self._model_name = model_name
        self._make_title()
        
        self.make_interactive = make_interactive
        
        self.size_attr = "size"
        
    # ===============
    # METHODS & UTILS
    # ===============
    
    def set_size_attr(self,
                      new_size_attr:str
                     ):
        assert new_size_attr in self._df_embeddings.columns
        
        self.size_attr = new_size_attr
        
        return
    
    def expand_df(self,
                  key,
                  values
                 ):
        self._df_embeddings[key] = values
        
        return
        
    
    def project_embs(self, inplace=True):
        
        embs = self._reducer.fit_transform(self._embeddings)
        
        if inplace:
            self._projected_embeddings = embs
            return 
        else:
            return embs
        
    def _make_df(self, inplace=True):
        if not hasattr(self, "_projected_embeddings"):
            self.project_embs()
            
        embx = [e[0] for e in self._projected_embeddings]
        emby = [e[1] for e in self._projected_embeddings]
        
        embs_df = pd.DataFrame([embx, emby], ["x", "y"]).T
        embs_df["size"] = 1 # TODO !
        embs_df["label"] = [str(lab) for lab in self._labels]
        embs_df["graphs"] = self._graphs
        
        if inplace:
            self._df_embeddings = embs_df
            return
        else:
            return embs_df
        
    def _make_title(self):
        
        title = ""
        
        if not self._dataset_name is None:
            title += "{} representation ".format(self._dataset_name)
            
        if not self._model_name is None:
            title += "computed with {} ".format(self._model_name)
        
        title += "projected from {input_dim}- to 2-dimensions with {red_type}.".format(input_dim=len(self._embeddings[0])
                                                                                       , red_type = self._reducer_type
                                                                                      )
                
        self.title = title
        return 
        
    # ================
    # PLOT & VISUALISE
    # ================
        
    def show_static(self,
                    **kwargs
                   ):
        static_viz = scatter_embedding_visualizer(
            df=self._df_embeddings
            , x_col="x"
            , y_col="y"
            , size_attr=self.size_attr
            , title=self.title
            , **kwargs
        )
        
        return static_viz
    
    def show_interactive(self,
                         **kwargs
                        ):
        go_viz = go_make_dash_embedding_visualizer(
            df=self._df_embeddings,
            x_col="x",
            y_col="y",
            size_attr="size",
            instance_plotter = instance_drawer_from_og_format(self._og_format),
            title=self.title,
            **kwargs
        )
        return go_viz
    
    
    def show(self,
             static_args:dict={},
             interactive_args:dict={}
            ):
        
        static_viz = self.show_static(
            **static_args
        )
        
        if self.make_interactive:
            interactive_viz = self.show_interactive(
                **interactive_args
            )
            
            return static_viz, interactive_viz
        
        return static_viz
            
            
            
            