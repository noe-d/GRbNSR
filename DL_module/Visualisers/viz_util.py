from .visualiser import *

import sklearn
from sklearn.manifold import TSNE
import pandas as pd

import dgl 
import networkx as nx
import graph_tool.all as gt
from pyintergraph import nx2gt

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, no_update
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash as Dash # tricks original package

import io
import base64
from PIL import Image


# ====================================================================================
#                                        utils                                        
# ====================================================================================

def rgb_to_hex(rgb):
    rgb = tuple(int(c*255) for c in rgb)
    return '#%02x%02x%02x' % rgb

def get_reducer_from_sklearn(module_path:str="manifold.TSNE"):
    """
    Example:
        get_reducer_from_sklearn("manifold.TSNE")

            loop f in ["manifold", "TSNE"]:
                red;sklearn-->sklearn.manifold-->sklearn.manifold.TSNE

            return sklearn.manifold.TSNE
    """
    red = sklearn
    for f in module_path.split("."):
        red = getattr(red, f)
        
    return (red, f)

def init_grviz_from_config(config):
    return 

# ====================================================================================
#                                static visualisation                                
# ====================================================================================

def scatter_embedding_visualizer(df:pd.DataFrame
                                 , x_col: str
                                 , y_col: str
                                 , size_attr: str
                                 , color_attr: str=None
                                 , cmap: str='viridis'
                                 , fig_dimensions:tuple = (25,15)
                                 , cat2color: dict=None
                                 , color_list: list=cm.get_cmap("tab10").colors
                                 , categorical_color:bool = False
                                 , title:str=None
                                 , save_path:str=None
                                 , do_save:bool=False
                                ):

    categorical_color = type(df[color_attr][0])==str
    
    fig, ax = plt.subplots(figsize=fig_dimensions)
    
    if not categorical_color:
        plot = sns.scatterplot(
            x=df[x_col],
            y=df[y_col],
            hue=df[color_attr],
            palette=cm.get_cmap(cmap),
            size=df[size_attr],
            alpha=0.4,
        )
        
        # add colorbar
        norm = plt.Normalize(df[color_attr].min(), df[color_attr].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig.colorbar(sm).outline.set_visible(False)
        
    else:
        
        if cat2color==None: 
            discrete_cats = sorted(df[color_attr].unique())
            cat2color= dict(zip(discrete_cats, color_list[:len(discrete_cats)]))
            
        plot = sns.scatterplot(
            x=df[x_col],
            y=df[y_col],
            hue=df[color_attr],
            palette=cat2color,
            size=df[size_attr],
            alpha=0.8,
        )
        
    # remove frame
    sns.despine(bottom = True, left = True)
    
    if title is not None:
        if len(title):
            plt.title(title, fontsize=14)

    plot.legend(fontsize=14)
    plt.tight_layout()
    # show plot
    if (not save_path is None) or (do_save):
        if save_path is None:
            save_path = "illustrations/scatter_embedding_visualisation.png"
        plt.savefig(save_path)
    else:
        plt.show()
    
    return


# ====================================================================================
#                                   Graph Plotters                                   
# ====================================================================================

def show_graph_tool(gt_name
                    , given_graph:bool=False
                    , **kwargs
                   ):
    network_name = gt_name
    if not given_graph:
        g = gt.collection.ns[network_name]
    else:
        g = gt_name

    ## fig, ax = plt.subplots()
    ## nx.draw(pyintergraph.gt2nx(g), ax=ax)

    # dump it to base64
    buffer = io.BytesIO()
    #im.save(buffer, format="jpeg")

    ## plt.savefig(buffer, format="jpeg")

    gt.graph_draw(g
                  , output=buffer
                  , fmt="png"
                  , **kwargs
                 )
    
    return buffer

_DEFAULT_NX_ARGS = {
    "node_size":5,
    "width":0.25,
    #"style":":",
    "alpha":0.8,
    "node_color":"#F77F00",
    "edge_color":"#0092E0",
}

import base64
def show_dgl_graph(dglgraph,
                   nx_draw_args:dict=_DEFAULT_NX_ARGS
                  ):
    """
    # convert 
    nxgraph = dgl.to_networkx(dglgraph).to_undirected()
    nx.draw(nxgraph, **nx_draw_args)

    buffer = io.BytesIO()
    #im.save(buffer, format="jpeg")
    plt.savefig(buffer, format="jpeg")
    plt.close()
    """
    gtgraph = nx2gt(dgl.to_networkx(dglgraph).to_undirected())
    buffer = show_graph_tool(
        gt_name = gtgraph
        , given_graph=True
        #, **kwargs
    )
    
    return buffer
    
    
def instance_drawer_from_og_format(og_format:str="DGL"):
    if og_format == "DGL":
        return show_dgl_graph
    elif og_format == "graph-tool":
        return show_graph_tool
    else:
        raise NotImplementedError

# ====================================================================================
#                              interactive visualisation                              
# ====================================================================================

def go_make_dash_embedding_visualizer(df:pd.DataFrame
                                      , x_col: str
                                      , y_col: str
                                      , size_attr: str
                                      , instance_plotter
                                      , instance_plotter_args:dict = {}
                                      , title:str = None
                                      , color_attr: str=None
                                      , cmap: str='viridis'
                                      , fig_dimensions:tuple = (1200, 1200)
                                      , cat2color: dict=None
                                      , color_list:list=cm.get_cmap("tab10").colors
                                      , min_size:int=None
                                     ):
    """
    Make the Dash application to visualize network embeddings 
        and explore the embedding space in an interactive fashion.
    
    """
    if color_attr == None:
        color_attr = size_attr
    if min_size is not None:
        df = df.copy()
        new_sizes = [s if s>min_size else min_size for s in df[size_attr]]
        df[size_attr] = new_sizes
        
    if not isinstance(color_list[0], str):
        color_list = [rgb_to_hex(c[:3]) for c in color_list]
        
    categorical_color = type(df[color_attr][0])==str
    
    # instatiatite plotly figure
    # --> if categorical color: treat special case
    if not categorical_color:
        fig = go.Figure(data=[
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    colorscale=cmap,
                    color=df[color_attr],
                    size=df[size_attr],
                    colorbar={"title": "Number of<br>{}".format(color_attr)},
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=45,
                    sizemode="area",#"diameter",
                    opacity=0.8,
                )
            )
        ])  
    else:
        
        if cat2color==None:
            discrete_cats = sorted(df[color_attr].unique())
            cat2color= dict(zip(discrete_cats, color_list[:len(discrete_cats)]))
        else:
            discrete_cats = list(cat2color.keys())
                
        fig = go.Figure()

        for c in discrete_cats:
            df_color= df[df[color_attr] == c]
            fig.add_trace(
                go.Scatter(
                    x=df_color[x_col], 
                    y=df_color[y_col],
                    name=c,
                    mode='markers',
                    #marker=dict(color=df_color[color_attr].map(cat2color))
                    marker=dict(
                        color=df_color[color_attr].map(cat2color),
                        size=df[size_attr],
                        line={"color": "#444"},
                        reversescale=True,
                        sizeref=45,
                        sizemode="area",#"diameter",
                        opacity=0.8,
                    ),
                    showlegend=True,
                )
            )

    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none"
                      , hovertemplate=None
                     )
    
    fig.update_layout(
        xaxis=dict(title='Dim 1'),
        yaxis=dict(title='Dim 2'),
        plot_bgcolor='rgba(255,255,255,0.1)',
        width=fig_dimensions[0],
        height=fig_dimensions[1],
    )
    
    # instatiate Dash application
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
        html.Button('Click here to sshUTDOWN', id='shutdown'),
    ])
    
    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        """
        Information / Viz.
            to be dsiplayed on hover.
        """

        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        curve_num = pt["curveNumber"]
        
        if categorical_color: # find the right trace!
            df_curve = df[df[color_attr] == discrete_cats[curve_num]]
            df_row = df_curve.iloc[num]
            
        else:
            df_row = df.iloc[num]

        ###########################################################
        ###########################################################

        #show_graph_tool(gt_name=df_row["Name"])
        network_name = df_row["graphs"]
        if not isinstance(network_name, str):
            network_name = "({}, {})".format(
                network_name.num_nodes(),
                network_name.num_edges()
            )
            
        buffer = io.BytesIO()
        buffer = instance_plotter(df_row["graphs"], **instance_plotter_args)
        
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image

        ###########################################################
        ###########################################################


        category = df_row['label']
        tags = ""#df_row['Tags']

        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "150px"},
                ),
                html.H2(f"{category}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
                html.P(f"{network_name}"),
                html.P(f"{tags}"),
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children
    
    from flask import request
    def shutdown_server():
        socketio2.stop()
        #func = request.environ.get('werkzeug.server.shutdown')
        #func = request.environ.get('Visualisers.viz_util.server.shutdown')
        #if func is None:
        #    raise RuntimeError('Not running with the Werkzeug Server')
        #func()

    @app.server.get('/shutdown')
    def shutdown():
        shutdown_server()
        return 'Server shutting down...'
    
    """
    @app.server.route('/shutdown', methods=['POST'])
    def shutdown():
        print("SHUTDOWN!!!!!!")
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
    """
        
    """
    @app.callback(
        Output("graph-basic-2", "figure"),
        Input("shutdown", "n_clicks"),
    )
    def shutdown(n_clicks):
        from flask import request
        import requests
        print("SHUUUTDOOOWN.......")
        host = "localhost"
        port = "8060"
        #shutdown_url = "http://{host}:{port}/_shutdown_{token}".format(
        #    host=host, port=port, token=Dash._token
        #)
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        shutdown_url = "http://{host}:{port}/_shutdown_{token}".format(
            host=host, port=port, token=Dash._token
        )
        try:
            response = requests.get(shutdown_url)
            print(response)
        except Exception as e:
            print(e)
    """

    #app.run_server(run_server_type)
    return app