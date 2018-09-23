#  Here we construct Lagrange polynomial basis, iterpolation and their derivatives

# lagrange polynomial basis of degree NP-1
def lagrange_basis(NP,i,x,xi):
    h = 1.;
    for j in range(1,NP+1):
        if (j != i): 
            num = x - xi[j-1]
            den = xi[i-1]-xi[j-1]
            h = h * num/den 
    return h

# lagrange polynomial interpolant of degree NP-1
def interpol(NP,x,xi,f):
    g = 0.
    for i in range(1,NP+1):
        a_i = lagrange_basis(NP,i,x,xi)
        g = g + a_i * f[i-1]
    return g

# derivatives of Lagrange polynomial basis of degree NP-1
def alternative_dl(j,x,z):
    y = 0.
    n = len(x)
    for l in range(1,n+1):
        if l != j:
            k = 1/(x[j-1]-x[l-1])
            for m in range(1,n+1):
                if (m!=j) and (m!=l):
                    k = k *(z-x[m-1])/(x[j-1]-x[m-1])
            y = y + k
    return y
