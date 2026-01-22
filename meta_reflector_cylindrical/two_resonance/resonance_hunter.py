import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *

"""
SCAN FOR ADDITIONAL RESONANCES
Fixed geometry, sliding wavelength window
Detect pole proximity via slope + curvature
"""

# =========================================================
# FIXED GEOMETRY (FROM YOUR SINGLE-RESONANCE DESIGN)
# =========================================================
params = [
    0.3035082512261301,  # r
    0.0745696719591017,  # h
    0.4613939870153822,   # hsio2
    0.7206415591170998    # a
]

# =========================================================
# GLOBAL SCAN RANGE
# =========================================================
lam_min = 1.20
lam_max = 2.10

window_width = 0.03
points_per_window = 12

window_centers = np.arange(
    lam_min + window_width/2,
    lam_max - window_width/2,
    window_width
)

# =========================================================
# MATERIAL MODEL ε(λ)
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "interp" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values
        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["interp"](wavelength)**2

# =========================================================
# CYLINDER GRID
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')
    x_c = x - a/2
    y_c = y - a/2
    mask = (x_c**2 + y_c**2) < r**2
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = et
    return ep

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1 = [a, 0]
    L2 = [0, a]
    es = epsilon_lambda(1/f)
    phi_inc = 0.0

    obj = grcwa.obj(nG, L1, L2, f, theta, phi_inc, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)

    for _ in range(5):
        obj.Add_LayerUniform(hs, es)
        obj.Add_LayerUniform(hsio2, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    epgrid = get_epgrids_cylinder(r, es, a).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0+nV]):
        return bi[k0]/ai[k0]
    else:
        return bi[k0+nV]/ai[k0+nV]

# =========================================================
# PHASE COMPUTATION
# =========================================================
def compute_phase(lams):
    r, h, hsio2, a = params
    phis = []
    for lam in lams:
        phis.append(np.angle(solver_system(1/lam, r, h, hsio2, a)))
    return np.unwrap(np.array(phis))

# =========================================================
# SLIDING WINDOW ANALYSIS
# =========================================================
results = []

for lc in window_centers:

    lams = np.linspace(
        lc - window_width/2,
        lc + window_width/2,
        points_per_window
    )

    phi = compute_phase(lams)

    # slope
    A, B = np.polyfit(lams, phi, 1)

    # curvature
    dphi = np.gradient(phi, lams)
    ddphi = np.gradient(dphi, lams)
    curvature = np.max(np.abs(ddphi))

    # linearity
    rms = np.sqrt(np.mean((phi - (A*lams + B))**2))

    score = abs(A) * curvature / (rms + 1e-6)

    results.append([lc, A, curvature, rms, score])

results = np.array(results)

# =========================================================
# SORT BY SCORE
# =========================================================
idx = np.argsort(results[:,4])[::-1]
results = results[idx]

print("\n===== TOP CANDIDATE WINDOWS =====")
for i in range(10):
    print(
        f"λc={results[i,0]:.3f}, "
        f"|A|={abs(results[i,1]):.1f}, "
        f"curv={results[i,2]:.2e}, "
        f"rms={results[i,3]:.2e}"
    )

# =========================================================
# VISUALIZE BEST WINDOW
# =========================================================
best_lc = results[0,0]
lams = np.linspace(best_lc-window_width/2, best_lc+window_width/2, 20)
phi = compute_phase(lams)

plt.figure(figsize=(6,4))
plt.plot(lams, phi, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title(f"Best candidate window around λ ≈ {best_lc:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()
