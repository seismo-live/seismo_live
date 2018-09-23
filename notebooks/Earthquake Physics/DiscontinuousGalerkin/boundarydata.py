# Left boundary
def BC0(u,v,r,rho,mu,U,V):
    #
    # we generate left boundary data
    import numpy as np

    # local wave speed and impedance
    cs = np.sqrt(mu/rho) 
    Zs = rho*cs

    # boundary forcing
    g = 1.0 * (0.5*Zs*(1.0-r)*U - 0.5*(1+r)*V)

    # outgoing characteristics that must be preserved
    p = 0.5 * (Zs*u + v)

    # hat-variables
    BCu = (1.0+r)/Zs * p + 1.0/Zs * g
    BCv = (1.0-r)*p - g
    
    return {'BCu':BCu,'BCv':BCv}

# Right boundary
def BCn(u,v,r,rho,mu,U,V):
    #
    # we generate right boundary data
    import numpy as np
    # local wave speed and impedance
    cs = np.sqrt(mu/rho)
    Zs = rho * cs

    # boundary forcing
    f = 1.0 * (0.5*Zs*(1.0-r)*U + 0.5 *(1.0+r)*V)

    # outgoing characteristics that must be preserved
    q = 0.5 * (Zs * u -v)

    # hat-variables
    BCu = (1.0+r)/Zs * q + 1.0/Zs * f
    BCv = -(1.0-r)*q + f
    
    return {'BCu':BCu,'BCv':BCv}
