# computes spectral difference approximation of the spatial derivative of a grid function u
def Dx(u,D,NP,num_element,dx):
    import numpy as np
    ux = np.zeros(len(u))
    
    for j in range (1,num_element+1):
        ux[(j-1)*NP:NP+(j-1)*NP] = 2/dx[j-1] *(D.dot(u[(j-1)*NP:NP+(j-1)*NP]))
    return ux
