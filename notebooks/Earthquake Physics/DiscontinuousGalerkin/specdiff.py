def derivative_matrix_GL(N, x, w):
    # compute derivatives of the basis functions at the quadrature nodes D_ij = phi'_i(x_j)
    import quadraturerules
    import Lagrange
    import numpy as np
    
    D = np.zeros([N+1,N+1])
    for i in range(1, N+2):
        for j in range(1,N+2):
            D[i-1,j-1] = Lagrange.alternative_dl(i,x,x[j-1])  # D_ij = phi'_i(x_j)
    return D

def derivative_GL(N, x, w):
    import quadraturerules
    import Lagrange
    import numpy as np
    # generate the spectral difference operator: D = W^(-1)*Q
    # with the stiffness matrix:  Q_ij = sum_m^N+1 (w_m*phi'_j(x_m))phi_i(x_m)) 
    
    D = np.zeros([N+1,N+1])
    phi_i = np.zeros([N+1,N+1])

    # Lagrange basis
    for i in range(1,N+2):
        for j in range(1,N+2):
            phi_i[i-1,j-1] = Lagrange.lagrange_basis(N+1,i,x[j-1],x)
            
    # derivativess of Lagrange bases        
    D0 = derivative_matrix_GL(N, x, w)

    # the spectral difference operator
    for i in range (1,N+2):
        for j in range(1, N+2):
            D[i-1,j-1] = 1.0 / w[j-1] * sum(np.multiply(np.multiply(np.transpose(w),D0[:,i-1]),phi_i[:,j-1]))
    return D




