import pandas as pd
import numpy as np
#import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

_COMMON_COLUMNS = [
    'gen',
    'best_fit',
    'best_geno_size',
    'lowest_fit',
    'lowest_fit_geno_size',
    'gen_comp_time',
    'sim_comp_time',
    'fit_comp_time',
]

"""
    helpers and loading and stuff
"""

def load_evo(path_to_evo:str="evo.csv"):
    return pd.read_csv(path_to_evo)

def get_df_changes(data:pd.DataFrame, 
                   key:str
                  ):
    assert key in data.columns, "{} must be one of the dataframe columns.".format(key)
    
    df_changes = data.copy()
    changes_mask = np.append(np.array([True]), (np.array(df_changes[1:][key]) != np.array(df_changes[:-1][key])))
    
    df_changes = df_changes[changes_mask]
    
    return df_changes
    

"""
    plotting
"""

def radar_stats_evo(ax,
                    data:pd.DataFrame,
                    stats:list,
                    colormap_name:str='copper',
                    labels:list=None,
                   ):
    if labels is None:
        labels = stats
        
    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # retrieve values to plot
    values = list(data.iloc[-1][stats].values)
    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values += values[:1]
    angles += angles[:1]
    
    cmap = get_cmap(colormap_name)
    
    
    last_id = data.index[-1]
    n_rows = len(data)

    for n_row, (id_row, row) in enumerate(data[1:].iterrows()):

        values = list(row[stats].values)
        values += values[:1]

        #color_row = "red" if id_row == last_id else "grey"
        color_row = cmap(n_row/n_rows)
        lw_row = 2 if id_row == last_id else 1
        alpha_row = (1+n_row)/(1+n_rows)
        ls_row = '-' if id_row == last_id else ':'

        # Draw the outline of our data.
        ax.plot(angles, values
                , color=color_row
                , linewidth=lw_row *2
                , alpha=alpha_row
                , ls=ls_row
               )

    # Fill it in.
    #ax.fill(angles, values, color='red', alpha=0.25)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)


    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)

    return ax


def loss_size_evo(ax,
                  data:pd.DataFrame,
                  loss_type:str,
                  data_changes:pd.DataFrame=None,
                  time_key:str=None,
                  size_key:str=None,
                  cmap_name:str='cool',
                  dilat_size:int=100,
                  threshold_break = 250,
                 ):
    
    if size_key is None:
        size_key = "best_geno_size"
    if time_key is None:
        time_key = 'gen'
    if data_changes is None:
        data_changes = get_df_changes(data=data, key=loss_type)
    
    for k in [loss_type, time_key, size_key]:
        assert k in data.columns, "{} must be in data columns.".format(size_key)
        assert k in data_changes.columns, "{} must be in data_changes columns.".format(size_key)
        
    ax.scatter(data_changes[time_key]
               , data_changes[loss_type]
               , c=data_changes[size_key]
               , s=dilat_size*data_changes[size_key]
               , cmap=cmap_name
              )

    plot_untill = int(data_changes.iloc[-1][time_key]+50)
    data_untill = data[:plot_untill]
    
    ax.plot(data[time_key][:plot_untill]
             , data[loss_type][:plot_untill]
             , ls=':'
             , c='b'
            )
    #plt.colorbar(ax=ax)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.semilogy()
    
    return ax



"""
    class encapsulation `EvoAnalysis`
"""
    
class EvoAnalysis(object):
    def __init__(
        self,
        path_to_evo_csv:str,
        evo_key:str = "best_fit",
        evo_key_size:str = "best_geno_size",
        stats_columns:list = ['DEGREES', 'U_PAGERANKS', 'TRIAD_CENSUS', 'U_DISTS'],
    ):
        # set params
        self.path_to_csv = path_to_evo_csv
        self.evo_key = evo_key
        self.stats_cols = stats_columns
        
        self.evo_key_size = evo_key_size
        
        # make DFs
        if not isinstance(self.path_to_csv, pd.DataFrame):
            self.evo_df = load_evo(
                path_to_evo = self.path_to_csv
            )
        else:
            self.evo_df = self.path_to_csv
            self.path_to_csv = None
            
        self.loss_df_change = get_df_changes(
            data = self.evo_df,
            key = self.evo_key
        )
        
    def run_analysis(
        self,
        return_fig=False,
        do_radar_plot = False,
        do_ls_plot = True
    ):
        
        fig = plt.figure(figsize=(18,9))
        
        ind_plot = 1
        if len(self.stats_cols) > 2:
            do_radar_plot = True
            ind_radar = ind_plot
            ind_plot += 1
            
        if do_ls_plot:
            ind_ls = ind_plot
            ind_plot += 2
            
        subplots = "{r}{c}".format(r=1,
                                   c=np.sum([do_radar_plot, do_ls_plot])+1,
                                  )
        subplots += "{}"
            
        i_plot = 1 
        if do_radar_plot:
            ax_radar = fig.add_subplot(int(subplots.format(ind_radar)), projection='polar')
            i_plot+=1
            ax_radar = radar_stats_evo(
                ax=ax_radar,
                data=self.loss_df_change,
                stats=self.stats_cols,
                colormap_name='copper'
                #labels:list=None,
            )
            
        if do_ls_plot:
            #ax_ls = fig.add_subplot(int(subplots.format(i_plot)))
            ax_ls = plt.subplot2grid((1, int(subplots[1]))
                                     , (0, ind_ls-1)
                                     , colspan=2
                                    )
            i_plot+=1
            
            ax_ls = loss_size_evo(
                ax=ax_ls,
                data=self.evo_df,
                loss_type=self.evo_key,
                data_changes = self.loss_df_change,
                time_key='gen',
                size_key=self.evo_key_size,
                cmap_name='cool',                
                dilat_size=100,
            )
            
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            return
        
        
"""
    main
"""

_DEFAULT_ARGS = {
    "path_to_evo_csv" : 'replication/2014/words.0/evo.csv',
    "evo_key" : 'best_fit',
    "evo_key_size" : 'best_geno_size',
    "stats_columns" : ['DEGREES', 'U_PAGERANKS', 'TRIAD_CENSUS', 'U_DISTS'],
}

def main(args=_DEFAULT_ARGS):
    
    evoanal = EvoAnalysis(
        path_to_evo_csv = args["path_to_evo_csv"],
        evo_key= args["evo_key"],
        evo_key_size = args["evo_key_size"],
        stats_columns= args["stats_columns"],
    )
    
    evoanal.run_analysis()
    
    return



if __name__ == "__main__":
    # parse argument & defaults or else....
    main(args={})
        