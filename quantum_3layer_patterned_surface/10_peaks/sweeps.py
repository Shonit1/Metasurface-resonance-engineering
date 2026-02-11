import numpy as np
import matplotlib.pyplot as plt
import grcwa
from config import *
import pandas as pd
from scipy.interpolate import interp1d

# =========================================================
# USER INPUT
# =========================================================
NPZ_FILE = "stage1_SUCCESS_min0p5_20260202_152328.npz"  # <-- your file
N_CAV = 7
DBR_PAIRS = 5

h_pattern = 0.30
hsio2_dbr = 0.2586

LAMBDA_MIN = 1.38
LAMBDA_MAX = 1.52
N_LAM = 500

# =========================================================
# LOAD STAGE-1 GEOMETRY
# =========================================================
data = np.load(NPZ_FILE)
p = data["params"]

# ---- decode params (THIS MATCHES STAGE-1 SCRIPT) ----
pattern = p[:9]
hs_list = p[9:9+N_CAV]
hsp_list = p[9+N_CAV:9+2*N_CAV]
hs_dbr  = p[9+2*N_CAV]
a       = p[9+2*N_CAV+1]
phi0    = p[9+2*N_CAV+2]

print("Loaded Stage-1 geometry:")
print(f"  pattern  = {pattern}")
print(f"  hs_list  = {hs_list}")
print(f"  hsp_list = {hsp_list}")
print(f"  hs_dbr   = {hs_dbr:.4f}")
print(f"  a        = {a:.4f}")
print(f"  phi0     = {phi0:.3f}")

# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0], data.iloc[:,1],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    n = 3.7 if wavelength < 1.0 else _cache["li"](wavelength)
    return n**2

# =========================================================
# 3×3 METASURFACE GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = np.array(pattern).reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair
    
    dx = dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            f = np.clip(pattern[i,j], 0, 1)
            ep[(X>=i*dx)&(X<(i+1)*dx)&
               (Y>=j*dy)&(Y<(j+1)*dy)] = f*eps + (1-f)*eair
    return ep

# =========================================================
# RCWA SOLVER (SAFE)
# =========================================================
def solve_rt(lam):
    
    eps_si = epsilon_lambda(lam)
      
    obj = grcwa.obj(
            nG, [a,0], [0,a],
            1/lam, theta, phi, verbose=0
        )
       
        # ---------- TOP ----------
    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h_pattern, Nx, Ny)
        

        # ---------- CAVITY STACK ----------
    for hs, hsp in zip(hs_list, hsp_list):
        obj.Add_LayerUniform(hsp, esio2)
        obj.Add_LayerUniform(hs, eps_si)
        
        # ---------- DBR (CRITICAL) ----------
    for _ in range(DBR_PAIRS):
        obj.Add_LayerUniform(hs_dbr, eps_si)
        obj.Add_LayerUniform(hsio2_dbr, esio2)
        
        # ---------- BOTTOM ----------
    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()
        
        # patterned layer eps
    ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
    obj.GridLayer_geteps(ep)

    obj.MakeExcitationPlanewave(1,0,0,0)
        
        # reflection
    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
    k0 = np.where((obj.G[:,0]==0) & (obj.G[:,1]==0))[0][0]
    nV = obj.nG

    r_raw = (
            bi[k0]/ai[k0]
            if abs(ai[k0]) > abs(ai[k0+nV])
            else bi[k0+nV]/ai[k0+nV]
        )
    print('r_raw:', r_raw)
    r = complex(r_raw)

    if not (np.isfinite(r.real) and np.isfinite(r.imag)):
            return None, None

        # transmission
    _, T = obj.RT_Solve(normalize=1, byorder=1)

    return r, float(T[k0])

    


# =========================================================
# WAVELENGTH SWEEP
# =========================================================
lams = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_LAM)

phis = []
R = []
T = []

for lam in lams:
    r, t = solve_rt(lam)
    if r is None:
        phis.append(np.nan)
        R.append(np.nan)
        T.append(np.nan)
        
    else:
        phis.append(np.angle(r))
        R.append(abs(r)**2)
        T.append(t)

phis = np.unwrap(phis) + phi0

# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(7,4))
plt.plot(lams, phis)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped reflection phase (rad)")
plt.title("Stage-1 geometry: reflection phase")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(lams, T, label="Transmission")
plt.plot(lams, R, label="Reflection")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Power")
plt.legend()
plt.title("Stage-1 geometry: T / R")
plt.grid(True)
plt.tight_layout()
plt.show()
