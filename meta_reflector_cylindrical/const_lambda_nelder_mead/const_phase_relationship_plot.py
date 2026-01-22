import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *

# =========================================================
# INSERT YOUR OPTIMIZED PARAMETERS HERE
# =========================================================
r_opt     = 0.233   # <-- replace with your value
h_opt     = 0.120
hsio2_opt = 0.300
a_opt     = 0.600

# =========================================================
# Wavelength range (µm)
# =========================================================
lambdas = np.linspace(1.45, 1.50, 25)

# =========================================================
# Material model ε(λ)
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
    n_val = _cache["interp"](wavelength)
    return n_val**2


# =========================================================
# Cylinder ε-grid
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')
    x_c = x - a / 2
    y_c = y - a / 2
    mask = (x_c**2 + y_c**2) < r**2
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair
    epgrid[mask] = et
    return epgrid


# =========================================================
# RCWA solver
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1 = [a, 0]
    L2 = [0, a]
    es = epsilon_lambda(1 / f)

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

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

    obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0, order=0)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


# =========================================================
# Compute UNWRAPPED phase
# =========================================================
phis = []
for lam in lambdas:
    f = 1 / lam
    r00 = solver_system(f, r_opt, h_opt, hsio2_opt, a_opt)
    phis.append(np.angle(r00))

phis = np.unwrap(np.array(phis))


# =========================================================
# Automatically fit optimal reference length L0
# =========================================================
x = 1 / lambdas
A = np.vstack([2 * np.pi * x]).T
L0_opt, *_ = np.linalg.lstsq(A, phis, rcond=None)
L0_opt = L0_opt[0]

phi_ref = 2 * np.pi * L0_opt / lambdas
phi_res = phis - phi_ref
C = np.mean(phi_res)


# =========================================================
# NUMERICAL METRICS (REPORT THESE)
# =========================================================
print("\n===== CONSTANT PHASE DIAGNOSTICS =====")
print("Optimal L0 (µm):", L0_opt)
print("Residual RMS (rad):", np.std(phi_res))
print("Peak-to-peak residual (rad):", phi_res.max() - phi_res.min())

dphi_dlambda = np.gradient(phi_res, lambdas)
print("Residual slope (rad/µm):", np.mean(dphi_dlambda))


# =========================================================
# PLOTS — THIS IS WHAT YOU SHOW
# =========================================================

# 1️⃣ Absolute unwrapped phase (will look linear — expected)
plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("Absolute reflection phase (contains propagation)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2️⃣ Reference-subtracted residual phase (THIS shows constant-λ behavior)
plt.figure(figsize=(6,4))
plt.plot(lambdas, phi_res, 'o-', label="Residual phase")
plt.plot(lambdas, C * np.ones_like(lambdas), '--', label="Constant fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Constant phase vs wavelength (after reference subtraction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3️⃣ Residual phase error only (optional but very clear)
plt.figure(figsize=(6,4))
plt.plot(lambdas, phi_res - C, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase error (rad)")
plt.title("Deviation from constant phase")
plt.grid(True)
plt.tight_layout()
plt.show()
