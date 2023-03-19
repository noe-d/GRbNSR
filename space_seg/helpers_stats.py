import numpy as np
from sklearn import metrics
from scipy.spatial import cKDTree as KDTree

import sys
sys.path.append('./DL_module')
from Utils.tasks import svc_classify

def str_to_metrics(str_metric:str):
    if str_metric == "silhouette":
        pairwise_metric = pairwise_silhouette
        overall_metric = metrics.silhouette_score
    elif str_metric == "KLDivergence":
        pairwise_metric = KLdivergence
        overall_metric = mean_KLdivergence
        
    else:
        raise NotImplementedError("Metric '{}' not implemented.".format(str_metric))
        
    return pairwise_metric, overall_metric
    
        

    
def pairwise_silhouette(x, y): # same format as KLdivergence
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)
    # ===================================
    
    xy_array = np.array(list(x)+list(y))
    proxy_labels = np.array([0]*len(x)+[1]*len(y))
    
    return metrics.silhouette_score(xy_array, proxy_labels)

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))