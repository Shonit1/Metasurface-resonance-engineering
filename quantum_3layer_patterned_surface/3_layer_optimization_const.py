import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import grcwa
from config import *

# =========================================================
# GEOMETRY (YOUR OPTIMIZED ONE)
# =========================================================
h1    = 0.284860
h2    = 0.083065
hs    = 0.298298
hsio2 = 0.278357

P1 = np.array([[0,0,0],[1,1,1],[0,0,0]])
P2 = np.array([[1,1,0],[0,1,1],[0,0,1]])
a_fixed = 1.4641

# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0].values,
            data.iloc[:,1].values,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["li"](wavelength)**2

# =========================================================
# 3×3 GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    ep = np.ones((Nx,Ny), dtype=complex) * eair
    dx = dy = a/3

    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[(X>=i*dx)&(X<(i+1)*dx)&
                   (Y>=j*dy)&(Y<(j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVERS
# =========================================================
def solver_rt(lam):
    eps_si = epsilon_lambda(lam)

    obj = grcwa.obj(
        nG, [a_fixed,0], [0,a_fixed],
        1/lam, theta, phi, verbose=0
    )

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h1, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)
    obj.Add_LayerGrid(h2, Nx, Ny)

    for _ in range(5):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsio2_dbr, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
    ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
    obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

    obj.MakeExcitationPlanewave(1,0,0,0)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    nV = obj.nG

    r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) else bi[k0+nV]/ai[k0+nV]
    t = Ti[k0]

    return r, t

# =========================================================
# STEP 1: COARSE SCAN (FIND TRUE RESONANCE)
# =========================================================
lams_coarse = np.linspace(1.498, 1.502, 4001)
R_coarse = []

for lam in lams_coarse:
    r, _ = solver_rt(lam)
    R_coarse.append(abs(r)**2)

R_coarse = np.array(R_coarse)
lam_res = lams_coarse[np.argmin(R_coarse)]

print(f"✓ Found resonance at λ = {lam_res*1000:.6f} nm")

# =========================================================
# STEP 2: ULTRA-FINE ZOOM (PICOMETER SCALE)
# =========================================================
dlam_pm = 20          # ±20 pm window
Npts    = 6001

dlam_um = dlam_pm * 1e-6
lams = np.linspace(lam_res-dlam_um, lam_res+dlam_um, Npts)

R = []
T = []
phi = []

for lam in lams:
    r, t = solver_rt(lam)
    R.append(abs(r)**2)
    T.append(np.real(t))
    phi.append(np.angle(r))

R = np.array(R)
T = np.array(T)
phi = np.unwrap(np.array(phi))

# Δλ in picometers (this is the key!)
delta_pm = (lams - lam_res) * 1e6
dphi_dlam = np.gradient(phi, delta_pm)

# =========================================================
# PLOTTING (THIS WILL LOOK BEAUTIFUL)
# =========================================================
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.plot(delta_pm, R)
plt.xlabel("Δλ (pm)")
plt.ylabel("Reflectance R")
plt.title("Reflectance Dip")
plt.grid(alpha=0.3)

plt.subplot(2,2,2)
plt.plot(delta_pm, T)
plt.xlabel("Δλ (pm)")
plt.ylabel("Transmittance T")
plt.title("Transmission Peak")
plt.grid(alpha=0.3)

plt.subplot(2,2,3)
plt.plot(delta_pm, phi)
plt.xlabel("Δλ (pm)")
plt.ylabel("Phase φ (rad)")
plt.title("Reflection Phase Jump")
plt.grid(alpha=0.3)

plt.subplot(2,2,4)
plt.plot(delta_pm, np.abs(dphi_dlam))
plt.xlabel("Δλ (pm)")
plt.ylabel("|dφ/dλ|  (rad / pm)")
plt.title("Phase Slope (Group Delay)")
plt.yscale("log")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("BIC_zoom_pm_all_observables.png", dpi=300)
plt.close()

print("✓ Saved: BIC_zoom_pm_all_observables.png")
