import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn import mixture
import sklearn.base as sklearn_base
from scipy import linalg

#ve_dict=ve_dict,
#ax = ax,
#title = title,
        
def visualise_kde(
    ve_dict,
    ax = None,
    title = None,
    legend=True,
    alpha=0.4,
    reducer=TSNE(),
):
    df_plot = pd.DataFrame()
    
    embs = np.array(ve_dict["embs"])
    
    if (not reducer is None) and (len(ve_dict["embs"][-1])!=2):
        embs = reducer.fit_transform(embs)
        

    df_plot["x0"] = embs[:,0]
    df_plot["x1"] = embs[:,1]
    df_plot["lab"] = ve_dict["labels"]

    sns.kdeplot(
        data = df_plot,
        x = "x0",
        y = "x1",
        hue="lab",
        fill=True,
        alpha=alpha,
        ax=ax,
        legend=legend
    )

    return ax

def visualise_gmm(
    ve_dict,
    ax = None,
    title = None,
    bayesian = False,
    show_pnts = False,
    color_list = sns.color_palette("colorblind"),
    #["sienna", "deepskyblue", "mediumorchid", "mediumseagreen", "lightcoral"],
    lims_dilat = 0.2,
    reducer=None,
):
    # ##############################
    def plot_unique_gmm(
        ax,
        mean,
        covar,
        color,
        alpha=0.4,
        reducer=TSNE()
    ):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean
                                  , v[0]
                                  , v[1]
                                  , angle=180.0 + angle
                                  , color=color
                                  , lw=0.
                                 )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ax.add_artist(ell)
        
        return ax
    # ##############################
    if ax is None:
        fig, ax = plt.subplots()
    
    unique_labels = np.unique(ve_dict["labels"])
    n_comps = len(unique_labels)
    
    colors = [color_list[i] for i in range(n_comps)]
    lab2color = dict(zip(unique_labels, colors))
    
    if bayesian:
        gmm = mixture.BayesianGaussianMixture(n_components=1#n_comps
                                              , covariance_type="full"
                                             )
    else:
        gmm = mixture.GaussianMixture(n_components=1#n_comps
                                      , covariance_type="full"
                                     )
        
    # fit
    #gmm.fit(ve_dict["embs"])
    
    embs = np.array(ve_dict["embs"])
    
    if (not reducer is None) and (len(ve_dict["embs"][-1])!=2):
        embs = reducer.fit_transform(embs)
    
    for lab in unique_labels:
        lab_gmm = sklearn_base.clone(gmm)
        
        lab_mask = np.array(ve_dict["labels"])==lab
        lab_embs = np.array(embs)[lab_mask]
        
        lab_gmm.fit(lab_embs)
        
        plot_unique_gmm(
            ax=ax,
            mean=lab_gmm.means_[0],
            covar=lab_gmm.covariances_[0],
            color=lab2color[lab]
        )
        
    if show_pnts:
        x1s = np.array(embs)[:,0]
        x2s = np.array(embs)[:,1]
        ax.scatter(x1s, x2s
                   , marker='x'
                   , c=[lab2color[lab] for lab in ve_dict["labels"]]
                   , alpha=0.6
                  )
        
    else:
        x_lims = (np.min(np.array(embs)[:,0]), np.max( np.array(embs)[:,0]))
        y_lims = (np.min(np.array(embs)[:,1]), np.max( np.array(embs)[:,1]))
        
        def dilat_lims(lims, dilat_coef = lims_dilat):
            dilat = (lims[0]-(lims[0]/np.abs(lims[0]))*dilat_coef*lims[0],
                     lims[1]+(lims[1]/np.abs(lims[1]))*dilat_coef*lims[1]
                    )
            return dilat
        
        if lims_dilat is not None and lims_dilat>0.:
            x_lims = dilat_lims(x_lims)
            y_lims = dilat_lims(y_lims)
        
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        
    return ax    