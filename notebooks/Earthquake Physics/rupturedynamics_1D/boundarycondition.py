import numpy as np

def bcm(mv, ms, v, s, rho, mu, r):
    V=0
    S=0
    cs = np.sqrt(mu/rho)
    zs = rho*cs

    p = 0.5*(zs*v - s)
    q = 0.5*(zs*v + s)
    
    P = 0.5*(zs*V - S)
    Q = 0.5*(zs*V + S) 
    
    g = P - r*Q

    mv[:] = 1.0/rho*(p - r*q - (P - r*Q))
    ms[:] = -mu/zs*(p - r*q - (P - r*Q))

def bcp(pv, ps, v, s, rho, mu, r):
    V=0
    S=0
    cs = np.sqrt(mu/rho)
    zs = rho*cs

    p = 0.5*(zs*v + s)
    q = 0.5*(zs*v - s)
    
    P = 0.5*(zs*V + S)
    Q = 0.5*(zs*V - S)
    
    g = P - r*Q

    pv[:] = 1.0/rho*(p - r*q - (P - r*Q))
    ps[:] = mu/zs*(p - r*q  - (P - r*Q))



