import numpy as np

def compute_rhat(x):
    # x shape expected: (nsteps, nwalkers, ndim)
    n, m = x.shape[0], x.shape[1]
    if n < 2: 
        return np.full(x.shape[-1], np.inf)
    mean_chain = np.mean(x, axis=0)
    mean_all = np.mean(mean_chain, axis=0)
    
    B = n / (m - 1) * np.sum((mean_chain - mean_all)**2, axis=0)
    W = 1 / (m * (n - 1)) * np.sum((x - mean_chain)**2, axis=(0, 1))
    
    # Prevent division by zero if a parameter hasn't moved yet
    W = np.where(W == 0, 1e-10, W) 
    
    var_plus = ((n - 1) / n) * W + B / n
    return np.sqrt(var_plus / W)