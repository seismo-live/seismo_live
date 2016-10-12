"""
Created on Wed Feb 17 18:00:41 2016
"""
import numpy as np

def flux(Q, N, ne, Ap, Am):
    """
    calculates the flux between two boundary sides of 
    connected elements for element i
    """
    # for every element we have 2 faces to other elements (left and right)
    out = np.zeros((ne,N+1,2))

    # Calculate Fluxes inside domain
    for i in range(1, ne-1):
        out[i,0,:] = Ap @ (-Q[i-1,N,:]) + Am @ (-Q[i,0,:])
        out[i,N,:] = Ap @ (Q[i,N,:])+ Am @ (Q[i+1,0,:])

    # Boundaries
    # Left 
    out[0,0,:] = Ap @ np.array([0,0]) + Am @ (-Q[i,0,:])
    out[0,N,:] = Ap @ (Q[0,N,:]) + Am @ (Q[1,0,:])

    # Right
    out[ne-1,0,:] = Ap @ (-Q[ne-2,N,:]) + Am @ (-Q[ne-1,0,:])
    out[ne-1,N,:] = Ap @ (Q[ne-1,N,:]) + Am @ np.array([0, 0])
    
    return out