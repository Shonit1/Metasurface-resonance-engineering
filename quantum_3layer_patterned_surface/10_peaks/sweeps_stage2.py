import numpy as np
import matplotlib.pyplot as plt
import grcwa
from config import *
import pandas as pd
from scipy.interpolate import interp1d

# =========================================================
# USER GEOMETRY (FROM YOUR RESULT)
# =========================================================

h_pattern = 0.2000
a         = 1.3451
hs_dbr    = 0.2200

hs_list = np.array([
    0.2200,
    0.2200,
    0.2679,
    0.2202,
    0.2330,
    0.2750,
    0.2200
])

hsp_list = np.array([
    0.1500,
    0.1500,
    0.1500,
    0.1500,
    0.1500,
    0.1500,
    0.1500
])

FIXED_PATTERN = np.array([
    [0.57213402, 0.49403910, 0.39748805],
    [0.23742280, 0.00000000, 0.00000000],
    [1.00000000, 0.00000000, 0.94079933]
])

# =========================================================
# RCWA SETTINGS (MATCH YOUR OPTIMIZATION)
# =========================================================



theta = 0.0
phi   = 0.0

DBR_PAIRS = 5

eair  = 1.0
esio2 = 1.44**2

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
    pattern = pattern
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
# RCWA SOLVER
# =========================================================

def solve_rt(lam):

    eps_si = epsilon_lambda(lam)

    obj = grcwa.obj(
        nG,
        [a, 0],
        [0, a],
        1 / lam,
        theta,
        phi,
        verbose=0
    )

    # Top air
    obj.Add_LayerUniform(0.1, eair)

    # Patterned layer
    obj.Add_LayerGrid(h_pattern, Nx, Ny)

    # Cavities
    for hs, hsp in zip(hs_list, hsp_list):
        obj.Add_LayerUniform(hsp, esio2)
        obj.Add_LayerUniform(hs, eps_si)

    # DBR
    for _ in range(DBR_PAIRS):
        obj.Add_LayerUniform(hs_dbr, eps_si)
        obj.Add_LayerUniform(hs_dbr, esio2)

    # Bottom air
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    ep = get_epgrid_3x3(FIXED_PATTERN, eps_si, a).flatten()
    obj.GridLayer_geteps(ep)

    obj.MakeExcitationPlanewave(1, 0, 0, 0)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)

    k0 = np.where((obj.G[:,0] == 0) & (obj.G[:,1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        r = bi[k0] / ai[k0]
    else:
        r = bi[k0 + nV] / ai[k0 + nV]

    r = complex(r)

    _, T = obj.RT_Solve(normalize=1, byorder=1)

    return r, T[k0]

# =========================================================
# WAVELENGTH SWEEP
# =========================================================

lam_min = 1.40
lam_max = 1.50
Nlam    = 500

lams = np.linspace(lam_min, lam_max, Nlam)

R = np.zeros(Nlam)
T = np.zeros(Nlam)
phi_r = np.zeros(Nlam)

for i, lam in enumerate(lams):
    r, t = solve_rt(lam)

    R[i] = np.abs(r)**2
    T[i] = t
    phi_r[i] = np.angle(r)

phi_r = np.unwrap(phi_r)

# =========================================================
# PLOTTING
# =========================================================

# =========================================================
# TARGET RESONANCE WAVELENGTHS (µm)
# =========================================================

lambda_targets = np.array([
    1.40,
    1.415,
    1.430,
    1.445,
    1.460,
    1.475,
    1.490
])





# =========================================================
# PLOTTING
# =========================================================

# ---------- PHASE PLOT ----------
plt.figure(figsize=(8, 6))
plt.plot(lams, phi_r, lw=2, label="Reflection phase")

for lam0 in lambda_targets:
    plt.axvline(
        lam0,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection Phase (rad)")
plt.title("Unwrapped Reflection Phase")
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------- POWER PLOT ----------
plt.figure(figsize=(8, 6))
#plt.plot(lams, R, label="Reflectance R", lw=2)
plt.plot(lams, T, label="Transmittance T", lw=2)
#plt.plot(lams, 1 - R - T, "--", label="Loss", lw=1)

for lam0 in lambda_targets:
    plt.axvline(
        lam0,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

plt.xlabel("Wavelength (µm)")
plt.ylabel("Power")
plt.title("Reflectance & Transmittance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
