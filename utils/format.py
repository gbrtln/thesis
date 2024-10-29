import numpy as np
import pandas as pd

def to_array(a):
    """
    Convert input data to numpy ndarray, where each row represents a feature.

    Parameters
    ----------
    a : array_like
        Statistical sample of data.
            
    Returns
    -------
    arr : ndarray
        Sample data formatted as numpy ndarray.
            
    See Also
    -------

    Notes
    ----

    References
    ---------

    Examples
    --------
    """  
    if isinstance(a, pd.Series):
        a = a.values
    elif isinstance(a, pd.DataFrame) and a.shape[1]==1:
        a = a.values.reshape(-1)
    elif isinstance(a, pd.DataFrame) and a.shape[1]>1:
        a = a.values.T
    # If input is a list or tuple, covert to numpy array
    arr = np.asarray(a)
    return arr
