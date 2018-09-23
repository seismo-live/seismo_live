def ADER_Wave_dG(u,v,D,NP,num_element,dx,w,x,t,r0,rn,dt,rho,mu):

    import rate
    import numpy as np
        
    wave_dG_return = rate.wave_dG(u,v,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t)
    Klu = wave_dG_return['Hu']
    Klv = wave_dG_return['Hv']
    
    fac = dt
    K3u = fac * Klu
    K3v = fac * Klv
    
    for k in range(1,NP+1): 
        fac = fac * (dt) /(k+1)
        wave_dG_return = rate.wave_dG(Klu,Klv,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t)
        Klu = wave_dG_return['Hu']
        Klv = wave_dG_return['Hv']
        
        K3u = K3u + fac * Klu
        K3v = K3v + fac * Klv

    Hu = u + K3u
    Hv = v + K3v

    return {'Hu':Hu,'Hv':Hv}



def ADER_Wave_1D_GL(u_1,v_1,u_2,v_2,S,psi,D,NP,num_element1,num_element2,dx1,dx2,w,x,t,r0,rn,dt,rho1,mu1,rho2,mu2,alpha,Tau_0,fric_law):
    
    import rate
    import numpy as np
    
    wave_1D_GL_return = rate.wave_1D_Friction(u_1,v_1,u_2,v_2,S,psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2, x,t,alpha,Tau_0,fric_law)
    Klu_1 = wave_1D_GL_return['Hu_1']
    Klv_1 = wave_1D_GL_return['Hv_1']
    Klu_2 = wave_1D_GL_return['Hu_2']
    Klv_2 = wave_1D_GL_return['Hv_2']
    Kld   = wave_1D_GL_return['H_d']
    Klpsi   = wave_1D_GL_return['H_psi']

    fac = dt
    K3u_1 = fac * Klu_1
    K3v_1 = fac * Klv_1
    K3u_2 = fac * Klu_2
    K3v_2 = fac * Klv_2
    K3d =  fac * Kld
    K3psi =  fac * Klpsi
    
    for k in range(1,NP+1): 
        fac = fac * (dt) /(k+1)
        wave_1D_GL_return = rate.wave_1D_Friction(Klu_1,Klv_1,Klu_2,Klv_2,S,psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2,x,t,alpha,Tau_0,fric_law)
        Klu_1 = wave_1D_GL_return['Hu_1']
        Klv_1 = wave_1D_GL_return['Hv_1']
        Klu_2 = wave_1D_GL_return['Hu_2']
        Klv_2 = wave_1D_GL_return['Hv_2']
        Kld   = wave_1D_GL_return['H_d']
        Klpsi = wave_1D_GL_return['H_psi']
    
        K3u_1 = K3u_1 + fac * Klu_1
        K3v_1 = K3v_1 + fac * Klv_1
        K3u_2 = K3u_2 + fac * Klu_2
        K3v_2 = K3v_2 + fac * Klv_2
        K3d   = K3d + fac * Kld
        K3psi = K3psi + fac * Klpsi

    Hu_1 = u_1 + K3u_1
    Hv_1 = v_1 + K3v_1
    Hu_2 = u_2 + K3u_2
    Hv_2 = v_2 + K3v_2
    H_d = S + K3d
    H_psi = psi + K3psi
    
    return {'Hu_1':Hu_1,'Hv_1':Hv_1,'Hu_2':Hu_2,'Hv_2':Hv_2, 'H_d':H_d, 'H_psi':H_psi}






def RK4_Wave_dG(u,v,D,NP,num_element,dx,w,x,t,r0,rn,dt,rho,mu):
    
    import rate
    import numpy as np
    
    
    K1u = np.zeros(len(u))
    K2u = np.zeros(len(u))
    K3u = np.zeros(len(u))
    K4u = np.zeros(len(u))
    
    K1v = np.zeros(len(v))
    K2v = np.zeros(len(v))
    K3v = np.zeros(len(v))
    K4v = np.zeros(len(v))
    
    
    #print(K1u)
    
    #return

    wave_dG_return = rate.wave_dG(u,v,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t)
    K1u = wave_dG_return['Hu']
    K1v = wave_dG_return['Hv']
    
    #Hu = u + dt*K1u
    #Hv = v + dt*K1v
    
    #return {'Hu':Hu,'Hv':Hv}
    
    
    wave_dG_return = rate.wave_dG(u+0.5*dt*K1u,v+0.5*dt*K1v,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t+0.5*dt)
    K2u = wave_dG_return['Hu']
    K2v = wave_dG_return['Hv']
   
    
    wave_dG_return = rate.wave_dG(u+0.5*dt*K2u,v+0.5*dt*K2v,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t+0.5*dt)
    K3u = wave_dG_return['Hu']
    K3v = wave_dG_return['Hv']
    
    
    wave_dG_return =rate.wave_dG(u+dt*K3u,v+dt*K3v,D,NP,num_element,dx,w,r0,rn,rho,mu,x,t+dt)
    K4u = wave_dG_return['Hu']
    K4v = wave_dG_return['Hv']
    
    
    Hu = u + dt/6*(K1u + 2*K2u + 2*K3u + K4u)
    Hv = v + dt/6*(K1v + 2*K2v + 2*K3v + K4v)
    
    return {'Hu':Hu,'Hv':Hv}



def RK4_Wave_1D_GL(u_1,v_1,u_2,v_2,S,psi,D,NP,num_element1,num_element2,dx1,dx2,w,x,t,r0,rn,dt,rho1,mu1,rho2,mu2,alpha,Tau_0,fric_law):
    
    import rate
    import numpy as np
    
    wave_1D_GL_return = rate.wave_1D_Friction(u_1,v_1,u_2,v_2,S,psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2, x,t,alpha,Tau_0,fric_law)
    K1u_1 = wave_1D_GL_return['Hu_1']
    K1v_1 = wave_1D_GL_return['Hv_1']
    K1u_2 = wave_1D_GL_return['Hu_2']
    K1v_2 = wave_1D_GL_return['Hv_2']
    K1S   = wave_1D_GL_return['H_d']
    K1psi   = wave_1D_GL_return['H_psi']
    
    wave_1D_GL_return = rate.wave_1D_Friction(u_1+0.5*dt*K1u_1,v_1+0.5*dt*K1v_1,u_2+0.5*dt*K1u_2,v_2+0.5*dt*K1v_2,S+0.5*dt*K1S,psi+0.5*dt*K1psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2, x,t,alpha,Tau_0,fric_law)
    K2u_1 = wave_1D_GL_return['Hu_1']
    K2v_1 = wave_1D_GL_return['Hv_1']
    K2u_2 = wave_1D_GL_return['Hu_2']
    K2v_2 = wave_1D_GL_return['Hv_2']
    K2S   = wave_1D_GL_return['H_d']
    K2psi   = wave_1D_GL_return['H_psi']
    
    
    wave_1D_GL_return = rate.wave_1D_Friction(u_1+0.5*dt*K2u_1,v_1+0.5*dt*K2v_1,u_2+0.5*dt*K2u_2,v_2+0.5*dt*K2v_2,S+0.5*dt*K2S,psi+0.5*dt*K2psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2, x,t,alpha,Tau_0,fric_law)
    K3u_1 = wave_1D_GL_return['Hu_1']
    K3v_1 = wave_1D_GL_return['Hv_1']
    K3u_2 = wave_1D_GL_return['Hu_2']
    K3v_2 = wave_1D_GL_return['Hv_2']
    K3S   = wave_1D_GL_return['H_d']
    K3psi   = wave_1D_GL_return['H_psi']
    
    
    wave_1D_GL_return = rate.wave_1D_Friction(u_1+dt*K3u_1,v_1+dt*K3v_1,u_2+dt*K3u_2,v_2+dt*K3v_2,S+dt*K3S,psi+dt*K3psi,D,NP,num_element1,num_element2,dx1,dx2,w,r0,rn,rho1,mu1,rho2,mu2, x,t,alpha,Tau_0,fric_law)
    K4u_1 = wave_1D_GL_return['Hu_1']
    K4v_1 = wave_1D_GL_return['Hv_1']
    K4u_2 = wave_1D_GL_return['Hu_2']
    K4v_2 = wave_1D_GL_return['Hv_2']
    K4S   = wave_1D_GL_return['H_d']
    K4psi   = wave_1D_GL_return['H_psi']
    
    
    
    
    
    Hu_1 = u_1 + dt/6*(K1u_1 + 2*K2u_1 + 2*K3u_1 + K4u_1)
    Hv_1 = v_1 + dt/6*(K1v_1 + 2*K2v_1 + 2*K3v_1 + K4v_1)
    Hu_2 = u_2 + dt/6*(K1u_2 + 2*K2u_2 + 2*K3u_2 + K4u_2)
    Hv_2 = v_2 + dt/6*(K1v_2 + 2*K2v_2 + 2*K3v_2 + K4v_2)
    
    H_d = S + dt/6*(K1S + 2*K2S + 2*K3S + K4S)
    H_psi = psi + dt/6*(K1psi + 2*K2psi + 2*K3psi + K4psi)
    
    return {'Hu_1':Hu_1,'Hv_1':Hv_1,'Hu_2':Hu_2,'Hv_2':Hv_2, 'H_d':H_d, 'H_psi':H_psi}








