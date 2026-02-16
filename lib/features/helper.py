import numpy as np

# ----------------------------
# Robust statistics helper
# ----------------------------
def robust_stats(x):
    """
    Compute median, IQR, and CV-robust (IQR/median) for a 1D array.
    Returns nan if input is empty.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    
    median = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    
    # CV-robust: IQR / median (avoid divide by zero)
    cvrobust = iqr / median if median != 0 else np.nan
    
    return median, iqr, cvrobust