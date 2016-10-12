"""
Created on Wed Feb 17 20:46:08 2016
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
        out[i,0,:] = Ap[i,:,:] @ (-Q[i-1,N,:]) + Am[i,:,:] @ (-Q[i,0,:])                 
        out[i,N,:] = Ap[i,:,:] @ Q[i,N,:] + Am[i,:,:] @ Q[i+1,0,:]
        
    # Boundaries
    # Left
    out[0,0,:] = Ap[0,:,:] @ np.array([0,0]) + Am[0,:,:] @ (-Q[i,0,:])        
    out[0,N,:] = Ap[0,:,:] @ Q[0,N,:] + Am[0,:,:] @ Q[1,0,:]

    # Right
    out[ne-1,0,:] = Ap[ne-1,:,:] @ (-Q[ne-2,N,:]) + Am[ne-1,:,:] @ -Q[ne-1,0,:]    
    out[ne-1,N,:] = Ap[ne-1,:,:] @ (Q[ne-1,N,:]) + Am[ne-1,:,:] @ np.array([0, 0])

    return out