import numpy as np
import matplotlib.pyplot as plt
import grcwa
from config import *



def get_epgrids(Nx,Ny):
    
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)  
    x, y = np.meshgrid(x0, y0, indexing='ij')

   

    epgrid = np.ones((Nx, Ny), dtype=complex) * 1.0
    
    
    return epgrid





def solver_system(nG,L1,L2,f,theta,phi,hs,hsio2,Nx,Ny,p,s):

    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose = 1)

    obj.Add_LayerUniform(0.1,eair)
    
    for i in range(5):
        obj.Add_LayerUniform(hs,es)
        obj.Add_LayerUniform(hsio2,esio2)

    obj.Add_LayerUniform(0.1,eair)

    obj.Init_Setup()

    planewave={'p_amp':p,'s_amp':s,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    

    Ri,Ti = obj.RT_Solve(normalize=1,byorder=1)
    Mi,Gi = obj.Solve_FieldFourier(which_layer=0, z_offset=0)

    return Ri,Ti,Mi,Gi,obj


Ri,Ti,Mi,Gi,obj = solver_system(nG,L1,L2,f,theta,phi,hs,hsio2,Nx,Ny,1,0)    
#print(Ri)
#print(np.sum(Ri),np.sum(Ti))
#print(obj.G)
print(obj.GetAmplitudes(which_layer=0,z_offset=0))
