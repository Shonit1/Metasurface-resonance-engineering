import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *





def epsilon_lambda(wavelength, _cache={}):
    import numpy as np
    

    if "interp" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values

        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    n_val = _cache["interp"](wavelength)
    return n_val**2









def get_epgrids_cylinder(r, et):
    """
    Build epsilon grid for a single cylindrical TiO2 meta-atom
    (circular cross-section) centered in the unit cell.
    
    r  : radius of cylinder (same units as L1, L2)
    et : permittivity of Si
    """

    # Coordinate grid
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')

    # Center coordinates
    x_centered = x - L1[0] / 2
    y_centered = y - L2[1] / 2

    # Circular mask (x^2 + y^2 < r^2)
    mask = (x_centered**2 + y_centered**2) < r**2

    # Epsilon grid
    epgrid = np.ones((Nx, Ny), dtype=complex) * 1.0  # background eps
    epgrid[mask] = et

    return epgrid






def solver_system(nG,L1,L2,f,theta,phi,hs,hsio2,Nx,Ny,p,s):

    es = epsilon_lambda(1/f)



    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose = 1)

    obj.Add_LayerUniform(0.1,eair)

    obj.Add_LayerGrid(h,Nx,Ny)

    obj.Add_LayerUniform(hsio2,esio2)
    
    for i in range(5):
        obj.Add_LayerUniform(hs,es)
        obj.Add_LayerUniform(hsio2,esio2)

    obj.Add_LayerUniform(0.1,eair)

    obj.Init_Setup()


    

    epgrid1 = get_epgrids_cylinder(r,es)
    epgrid = epgrid1.flatten()

    obj.GridLayer_geteps(epgrid)



    planewave={'p_amp':p,'s_amp':s,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    

    Ri,Ti = obj.RT_Solve(normalize=1,byorder=1)
    Mi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    return Ri,Ti,Mi,obj


Ri,Ti,Mi,obj = solver_system(nG,L1,L2,f,theta,phi,hs,hsio2,Nx,Ny,1,0)    
print(Ri)
print(np.sum(Ri),np.sum(Ti))
#print(obj.G)
print(obj.GetAmplitudes(which_layer=0,z_offset=0))
print(obj.nG)