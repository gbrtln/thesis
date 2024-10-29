# © Lorenzo Cederle 2024

from . import fleishman
from . import format
from . import matrix
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union
import warnings


def mvsk(a, ddof=1, bias=False, axis=None, shrink_method=None, shrinkage=.1, to_tuple=False) -> Union[dict, tuple]:
    """
    Compute mean, variance, skewness, and (excess) kurtosis of a 
    given sample of data.
    
    If the input dimension is greater than one, compute also 
    covariance and correlation matrices.

    Parameters
    ----------
    a : array_like
        Statistical sample of data.
    ddof : int
       Degrees of freedom to estimate variance. Input value for
       scipy.stats.tvar.
    bias : bool
        Bias to estimate skewness and kurtosis. Input value for
        scipy.stats.skew and scipy.stats.kurtosis.
    axis : int (0, 1)
        Axis to compute statistics along.
    shrink_method : string
        Shrink method to apply.
    shrinkage : float
        Shrinkage parameter.
    to_tuple : bool
        If True, return output formatted as a tuple.
            
    Returns
    -------
    mvsk_stats : dict, tuple
        Dictionary containing the statistics computed on the input 
        data sample.
        {
            'mean': mean
         [, 'cov' covariance]
          , 'var': variance
         [, 'cov' covariance]
          , 'skew': skewness
          , 'ekurt': (excess) kurtosis
        }
        If parameter to_tuple is True, output is converted to a tuple 
        with the same ordering.
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """
    # Convert input to numpy ndarray of shape (n_variables, n_samples)
    a = format.to_array(a)
    
    # Compute statistics depending on the size of input
    if a.ndim == 1:
        mvsk_stats = {
            'mean': stats.tmean(a, axis=axis)
          , 'var': stats.tvar(a, ddof=ddof)
          , 'skew': stats.skew(a, bias=bias)
          , 'ekurt': stats.kurtosis(a, bias=bias)
        }
    else:
        # Compute covariance matrix. If the numpy Maximum Likelihood estimate is not 
        # positive-definite, use scikit-learn basic shrinkage method to provide a 
        # better estimate.
        # TO-DO: implement Ledoit-Wolf shrinkage and Oracle approximating shrinkage
        # TO-DO: consider implementing the following:
        # - Sparse inverse covariance
        # - Robust Covariance Estimation
        cov = np.cov(a, ddof=ddof)
        if shrink_method is not None:
            cov = matrix.shrink_matrix(cov, method=shrink_method, shrinkage=shrinkage)
        var = np.diag(cov)
        if not np.all(var) > 0.:
            raise ValueError('Not all variances are positive. Try using a shrinkage method or change shrinking parameter.')

        # Compute correlation matrix. Same methodology as for the covariance matrix.
        corr = np.corrcoef(a)
        if shrink_method is not None:
            corr = matrix.shrink_matrix(corr, is_corr=True, shrinkage=shrinkage)

        mvsk_stats = {
            'mean': stats.tmean(a, axis=axis)
          , 'cov': cov
          , 'var': var
          , 'corr': corr
          , 'skew': stats.skew(a, bias=bias, axis=axis)
          , 'ekurt': stats.kurtosis(a, bias=bias, axis=axis)
        }

    if to_tuple:
        mvsk_stats = tuple(mvsk_stats.values())
    return mvsk_stats


def mvsk_describe(df: pd.DataFrame, ddof=1, bias=False, pivot=False) -> pd.DataFrame:
    """
    Compute mean, variance, skewness, and (excess) kurtosis of the 
    columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        Statistical sample of data.
    ddof : int
       Degrees of freedom to estimate variance. Input value for
       scipy.stats.tvar.
    bias : bool
        Bias to estimate skewness and kurtosis. Input value for
        scipy.stats.skew and scipy.stats.kurtosis.
    pivot : bool
        If True, transpose the output.
            
    Returns
    -------
    mvsk_df : pandas DataFrame
        Pandas DataFrame containing mean, variance, skewness, and (excess) 
        kurtosis of the columns of the input DataFrame. If pivot = False, 
        each row represents a column of the input DataFrame. 
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """
    # Check input type
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input data must be formatted as a pandas DataFrame.')

    # Compute statistics for each column
    mean = pd.DataFrame(df.apply(lambda x: stats.tmean(x)), columns=['mean'])
    var = pd.DataFrame(df.apply(lambda x: stats.tvar(x, ddof=ddof)), columns=['var'])
    skew = pd.DataFrame(df.apply(lambda x: stats.skew(x, bias=bias)), columns=['skew'])
    ekurt = pd.DataFrame(df.apply(lambda x: stats.kurtosis(x, bias=bias)), columns=['ekurt'])

    mvsk_df = pd.concat([mean, var, skew, ekurt], axis=1)

    # Pivot the result
    if pivot:
        mvsk_df = mvsk_df.T
    return mvsk_df


def mvsk_compare(orig: pd.DataFrame, synth: pd.DataFrame, side_by_side=False, ddof=1, bias=False, pivot=False) -> pd.DataFrame:
    """
    """
    # Check if the input DataFrames have the same columns in the same order
    if not orig.columns.tolist() == synth.columns.tolist():
        raise ValueError('Input DataFrames must have the same column names in the same order.')
    mvsk_orig = mvsk_describe(orig, ddof=ddof, bias=bias, pivot=pivot)
    mvsk_synth = mvsk_describe(synth, ddof=ddof, bias=bias, pivot=pivot)
    mvsk = pd.concat([mvsk_orig, mvsk_synth], axis=1)
    # Create multi-index
    if side_by_side:
        cols = mvsk.columns.tolist()
        cols = [0, 4, 1, 5, 2, 6, 3, 7]
        mvsk = mvsk.iloc[:, cols]
        multi_index = pd.MultiIndex.from_tuples([(col, src) for col in mvsk_orig.columns for src in ['Original', 'Synthetic']])
        mvsk.columns = multi_index
    else:
        multi_index = pd.MultiIndex.from_tuples([(src, col) for src in ['Original', 'Synthetic'] for col in mvsk_orig.columns])
        mvsk.columns = multi_index
    return mvsk


def fl_coeff(a) -> dict:
    """
    Compute Fleishman's coefficients of a given statistical sample.

    Parameters
    ----------
    a : array_like, dict
        If array-like, the input is the statistical sample of interest.
        If dictionary, it must contain unbiased mean, variance, skewness, 
        and (excess) kurtosis of the statistical sample. In this case the
        input must be formatted as follows:
        {
            'mean': mean
          , 'var': variance
          , 'skew': skewness
          , 'ekurt': (excess) kurtosis
        }
            
    Returns
    -------
    ff_coeff : dict
        Dictionary containing coefficients of the Fleishman's polynomial 
        Y = a + b*X + c*X**2 + d*X**3, formatted as follows
        {
            'a': a
          , 'b': b
          , 'c': c
          , 'd': d
        }
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """
    # Check input type
    # TO-DO: enhance type check!
    if not isinstance(a, dict):
        a = np.array(a)
        if not isinstance(a, np.ndarray):
            raise TypeError('Input must be either a numpy array or a dictionary.')
        a = mvsk(a)

    # Initialize Fleishman instance and compute Fleishman's coefficients
    ff = fleishman.Fleishman(mean=a['mean'], var=a['var'], skew=a['skew'], ekurt=a['ekurt'])
    ff.fl_coeff()
    ff_coeff = ff.coeff
    return ff_coeff


def interm_corr(flc, corr, thrs=1e-10):
    """
    Compute intermediate correlation for Vale & Maurelli transformation, 
    given the coeffients of two Fleishman polynomials as input.

    Parameters
    ----------
    flc : tuple
        Tuple containing two dictionaries, which contain the Fleishman 
        coefficients of the two variables to be simulated. Each dictionary  
        must be formatted as follows:
        {
            'a': a
          , 'b': b
          , 'c': c
          , 'd': d
        }
    corr : float
        Original correlation of the two variables to be simulated.
    thrs : float
        Threshold under which the imaginary part of the roots of the 
        transformation polynomial is to be dropped.
            
    Returns
    -------
    ro : float
        Intermediate correlation for Vale & Maurelli transformation.
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """
    # Extract Fleishman's coefficients
    b1, c1, d1 = flc[0]['b'], flc[0]['c'], flc[0]['d']
    b2, c2, d2 = flc[1]['b'], flc[1]['c'], flc[1]['d']

    # Define polynomial whose root is the intermediate correlation
    p = [
        6.*d1*d2
      , 2.*c1*c2
      , b1*b2 + 3.*b1*d2 + 3.*d1*b2 + 9.*d1*d2
      , -1.*corr
    ]

    # Find all real roots up to a specified threshold
    # TO-DO: find a better method to manage the threshold parameter
    ro = np.roots(p)
    ro = ro.real[abs(ro.imag) < thrs]
    ro = [r for r in ro if np.abs(r) <= 1.]

    # Check result
    # TO-DO: find a better method to manage results. In particular
    # evaluate the effectiveness of the numerical solution
    if len(ro) == 0: # If no valid solution exist, use default correlation
        warnings.warn('No valid solution to compute intermediate correlation exists. Using original correlation as default.')
        ro = corr
    else:
        if len(ro) > 1: # If there are more real roots, warn the user and take the first one
            warnings.warn('There are more than one real roots with abs less than one.')
        ro = min(ro)
    return ro


def multivariate_non_normal(mean, cov, skew, ekurt, size=1, random_state=None):
    """
    Generate a multivariate sample of correlated non-normal variables, 
    using Vale & Maurelli method.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    skew : 1-D array_like, of length N
        Skewness of the N-dimensional distribution.
    ekurt : 1-D array_like, of length N
        Excess kurtosis of the N-dimensional distribution.
    size : int
        Number of samples to be generated.
    random_state : int
        Internal seed for the random generation of normal samples.
            
    Returns
    -------
    Y : ndarray
        Multivariate correlated non-normal samples of shape (N, size)
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """
    nvar=len(mean)

    # TO-DO: Check input type
    # TO-DO: Check input shape

    # Check semi-definite structure of covariance matrix
    if not matrix.isPD(cov):
        raise ValueError('Input covariance matrix must be positive-definite.')
    
    # Compute correlation matrix
    var = np.diag(cov)
    std = np.sqrt(var)
    corr = cov / np.outer(std, std)

    # If the correlation matrix is not positive semi-definite, 
    # redefine it as the nearest positive semi-definite matrix
    if not matrix.isPD(corr):
        corr = matrix.nearestPD(corr)

    # WIP: Once here, all inputs must be numpy arrays with correct dimension and shape

    # Compute Fleishman coefficients for each variable
    fl_coeff = []
    for i in range(nvar):
        ff = fleishman.Fleishman(mean=mean[i], var=var[i], skew=skew[i], ekurt=ekurt[i])
        ff.fl_coeff()
        fl_coeff.append(ff.coeff)

    # Compute intermediate correlation
    int_corr = np.ones((nvar, nvar))
    for i in range(nvar):
        for j in range(i+1, nvar):
            int_corr[i, j] = interm_corr((fl_coeff[i], fl_coeff[j]), corr[i, j])
            int_corr[j, i] = int_corr[i, j]
    
    # If the intermediate correlation matrix is not positive semi-definite, 
    # redefine it as the nearest positive semi-definite matrix
    if not matrix.isPD(int_corr):
        int_corr = matrix.nearestPD(int_corr)

    # Generate multivariate normal samples with intermediate correlation
    multvar_norm_array = stats.multivariate_normal.rvs(
        mean=np.zeros((nvar,))
      , cov=int_corr
      , size=size
      , random_state=random_state
    )
   
   # Compute Fleishman variables
    Y = np.zeros((nvar, size))
    for i in range(nvar):
        N = multvar_norm_array[:, i]
        b, c, d = fl_coeff[i]['b'], fl_coeff[i]['c'], fl_coeff[i]['d']
        Y[i, :] = (-1 * c + N * (b + N * (c + N * d))) * std[i] + mean[i]

    return Y
