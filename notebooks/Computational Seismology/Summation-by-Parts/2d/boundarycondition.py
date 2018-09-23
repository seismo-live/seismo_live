import numpy as np

def bcm(mv, ms, v, s, rho, mu, r):

    cs = np.sqrt(mu/rho)
    zs = rho*cs

    p = 0.5*(zs*v - s)
    q = 0.5*(zs*v + s)

    mv[:] = 1.0/rho*(p - r*q)
    ms[:] = -mu/zs*(p - r*q)

def bcp(pv, ps, v, s, rho, mu, r):

    cs = np.sqrt(mu/rho)
    zs = rho*cs

    p = 0.5*(zs*v + s)
    q = 0.5*(zs*v - s)

    pv[:] = 1.0/rho*(p - r*q)
    ps[:] = -mu/zs*(p - r*q)



def bcm2dx(BF, F, Mat, nx, ny, r, physics):

    # reflection coefficient on the right boundary
    r0x = r[0]

    if (physics == 'elastic'):
        # loop through all points on the boundary
        for j in range(0, ny):
            
            # extract material parameters
            rho = Mat[0, j, 0]
            Lambda = Mat[0, j, 1]
            mu = Mat[0, j, 2]
            
            # compute wave speeds
            cp = np.sqrt((2.0*mu + Lambda)/rho)
            cs = np.sqrt(mu/rho)
            
            # compute impedances
            zp = rho*cp
            zs = rho*cs
            
            # extract boundary fields
            vx = F[0, j, 0]
            vy = F[0, j, 1]
            sxx = F[0, j, 2]
            syy = F[0, j, 3]
            sxy = F[0, j, 4]

            # compute characteristics
            px = 0.5*(zp*vx - sxx)
            py = 0.5*(zs*vy - sxy)
            qx = 0.5*(zp*vx + sxx)
            qy = 0.5*(zs*vy + sxy)
    
            # compute SAT terms to be used in imposing bou
            BF[j,0] = 1.0/rho*(px - r0x*qx)
            BF[j,1] = 1.0/rho*(py - r0x*qy)
            BF[j,2] = -(2.0*mu + Lambda)/zp*(px - r0x*qx)
            BF[j,3] = -Lambda/zp*(px - r0x*qx)
            BF[j,4] =  -mu/zs*(py - r0x*qy)

    if (physics == 'acoustics'):
        # loop through all points on the boundary                                                                                      
        for j in range(0, ny):

            # extract material parameters                                                                                              
            rho = Mat[0, j, 0]
            lam = Mat[0, j, 1]
        
            # compute wave speeds                                                                                                     
            c = np.sqrt(lam/rho)

            # compute impedances                                                                                                     
            z = rho*c
             

            # extract boundary fields
            p  = F[0, j, 0]             
            vx = F[0, j, 1]
            vy = F[0, j, 2]

            # compute characteristics
            px = 0.5*(z*vx - p)
            qx = 0.5*(z*vx + p)

            # compute SAT terms to be used in imposing bou                                                                           
            BF[j,0] = -lam/z*(px - r0x*qx)
            BF[j,1] = 1.0/rho*(px - r0x*qx)
            BF[j,2] = 0.0
            

    

def bcp2dx(BF, F, Mat, nx, ny, r, physics):
    
    # reflection coefficient on the right boundary
    rnx = r[1]

    if (physics == 'elastic'):
        # loop through all points on the boundary
        for j in range(0, ny):
        
            # extract material parameters
            rho = Mat[nx-1, j, 0]
            Lambda = Mat[nx-1, j, 1]
            mu = Mat[nx-1, j, 2]

            # compute wave speeds
            cp = np.sqrt((2.0*mu + Lambda)/rho)
            cs = np.sqrt(mu/rho)

            # compute impedances
            zp = rho*cp
            zs = rho*cs
            
            # extract boundary fields
            vx = F[nx-1, j, 0]
            vy = F[nx-1, j, 1]
            sxx = F[nx-1, j, 2]
            syy = F[nx-1, j, 3]
            sxy = F[nx-1, j, 4]

            # compute characteristics
            px = 0.5*(zp*vx + sxx)
            py = 0.5*(zs*vy + sxy)
            qx = 0.5*(zp*vx - sxx)
            qy = 0.5*(zs*vy - sxy)
    
            # compute SAT terms to be used in imposing bou
            BF[j,0] = 1.0/rho*(px - rnx*qx)
            BF[j,1] = 1.0/rho*(py - rnx*qy)
            BF[j,2] = (2.0*mu + Lambda)/zp*(px - rnx*qx)
            BF[j,3] = Lambda/zp*(px - rnx*qx)
            BF[j,4] =  mu/zs*(py - rnx*qy)

    if (physics == 'acoustics'):
        
        # loop through all points on the boundary  
        for j in range(0, ny):
            # extract material parameters                                                                                            

            rho = Mat[nx-1, j, 0]
            lam = Mat[nx-1, j, 1]

            # compute wave speeds                                                                                                     
            c = np.sqrt(lam/rho)

            # compute impedances                                                                                                       
            z = rho*c


            # extract boundary fields                                                                                                  
            p  = F[nx-1, j, 0]
            vx = F[nx-1, j, 1]
            vy = F[nx-1, j, 2]

            # compute characteristics                                                                                                  
            px = 0.5*(z*vx + p)
            qx = 0.5*(z*vx - p)

            # compute SAT terms to be used in imposing bou                                                                             
            BF[j,0] = (lam/z)*(px - rnx*qx)
            BF[j,1] = (1.0/rho)*(px - rnx*qx)
            BF[j,2] = 0.0


def bcm2dy(BF, F, Mat, nx, ny, r, physics):

    # reflection coefficient on the right boundary
    r0y = r[2]

    if (physics == 'elastic'):
        # loop through all points on the boundary
        for i in range(0, nx):
            
            # extract material parameters
            rho = Mat[i, 0, 0]
            Lambda = Mat[i, 0, 1]
            mu = Mat[i, 0, 2]
            
            # compute wave speeds
            cp = np.sqrt((2.0*mu + Lambda)/rho)
            cs = np.sqrt(mu/rho)

            # compute impedances
            zp = rho*cp
            zs = rho*cs

            # extract boundary fields
            vx = F[i, 0, 0]
            vy = F[i, 0, 1]
            sxx = F[i, 0, 2]
            syy = F[i, 0, 3]
            sxy = F[i, 0, 4]

            # compute characteristics
            px = 0.5*(zs*vx - sxy)
            py = 0.5*(zp*vy - syy)
            qx = 0.5*(zs*vx + sxy)
            qy = 0.5*(zp*vy + syy)
    
            # compute SAT terms to be used in imposing bou
            BF[i,0] = 1.0/rho*(px - r0y*qx)
            BF[i,1] = 1.0/rho*(py - r0y*qy)
            BF[i,2] = -Lambda/zp*(py - r0y*qy)
            BF[i,3] = -(2.0*mu+Lambda)/zp*(py - r0y*qy)
            BF[i,4] =  -mu/zs*(px - r0y*qx)

    if (physics == 'acoustics'):
        # loop through all points on the boundary                                                                                      
        for j in range(0, nx):
            # extract material parameters                                                                                              
            rho = Mat[j, 0, 0]
            lam = Mat[j, 0, 1]

            # compute wave speeds                                                                                                     
            c = np.sqrt(lam/rho)

            # compute impedances                                                                                                       
            z = rho*c

            # extract boundary fields                                                                                                  
            p  = F[j, 0, 0]
            vx = F[j, 0, 1]
            vy = F[j, 0, 2]

            # compute characteristics                                                                                                  
            py = 0.5*(z*vy - p)
            qy = 0.5*(z*vy + p)

            # compute SAT terms to be used in imposing bou                                                                             
            BF[j,0] = -lam/z*(py - r0y*qy)
            BF[j,1] = 0.0
            BF[j,2] = 1.0/rho*(py - r0y*qy)
            


def bcp2dy(BF, F, Mat, nx, ny, r, physics):

    # reflection coefficient on the right boundary
    rny = r[3]

    if (physics == 'elastic'):
        # loop through all points on the boundary
        for i in range(0, nx):
            
            # extract material parameters
            rho = Mat[i, ny-1, 0]
            Lambda = Mat[i, ny-1, 1]
            mu = Mat[i, ny-1, 2]

            # compute wave speeds
            cp = np.sqrt((2.0*mu + Lambda)/rho)
            cs = np.sqrt(mu/rho)

            # compute impedances
            zp = rho*cp
            zs = rho*cs

            # extract boundary fields
            vx = F[i, ny-1, 0]
            vy = F[i, ny-1, 1]
            sxx = F[i, ny-1, 2]
            syy = F[i, ny-1, 3]
            sxy = F[i, ny-1, 4]

            # compute characteristics
            px = 0.5*(zs*vx + sxy)
            py = 0.5*(zp*vy + syy)
            qx = 0.5*(zs*vx - sxy)
            qy = 0.5*(zp*vy - syy)
    
            # compute SAT terms to be used in imposing boundary conditions
            BF[i,0] = 1.0/rho*(px - rny*qx)
            BF[i,1] = 1.0/rho*(py - rny*qy)
            BF[i,2] = Lambda/zp*(py - rny*qy)
            BF[i,3] = (2.0*mu+Lambda)/zp*(py - rny*qy)
            BF[i,4] = mu/zs*(px - rny*qx)

    if (physics == 'acoustics'):
        # loop through all points on the boundary
        for i in range(0, nx):
            # extract material parameters 
            rho = Mat[i, ny-1, 0]
            lam = Mat[i, ny-1, 1]

            # compute wave speeds                                                                                                     
            c = np.sqrt(lam/rho)

            # compute impedances
            z = rho*c

            # extract boundary fields
            p  = F[i, ny-1, 0]
            vx = F[i, ny-1, 1]
            vy = F[i, ny-1, 2]

            # compute characteristics
            py = 0.5*(z*vy + p)
            qy = 0.5*(z*vy - p)

            # compute SAT terms to be used in imposing bou 
            BF[i,0] = (lam/z)*(py - rny*qy)
            BF[i,1] = 0.0
            BF[i,2] = (1.0/rho)*(py - rny*qy)
    
