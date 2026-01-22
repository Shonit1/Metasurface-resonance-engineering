import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *


'''
After finding the resonance slope, this zeroes in on the resonance and fits the sqrt behaviour.

'''


# =========================================================
# TARGET POLE LOCATION
# =========================================================
lambda0 = 1.526  # µm

# =========================================================
# WAVELENGTH GRID (DENSE NEAR POLE)
# =========================================================
lambdas = np.concatenate([
    np.linspace(1.505, 1.522, 6),
    np.linspace(1.522, 1.530, 14),  # dense near pole
    np.linspace(1.530, 1.545, 6)
])

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
    n_val = _cache["interp"](wavelength)
    return n_val**2

# =========================================================
# CYLINDER ε-GRID
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
# RCWA SOLVER
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

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]

# =========================================================
# PHASE COMPUTATION
# =========================================================
def compute_phase(params):
    r, h, hsio2, a = params
    phis = []
    for lam in lambdas:
        f = 1 / lam
        r00 = solver_system(f, r, h, hsio2, a)
        phis.append(np.angle(r00))
    return np.unwrap(np.array(phis))

# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array([
    [0.25, 0.35],   # r
    [0.06, 0.3],   # h
    [0.4, 0.65],   # hsio2
    [0.60, 0.8]    # a
])

def decode(x):
    x = np.clip(x, 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

# =========================================================
# TARGET SQRT FUNCTION
# =========================================================
def sqrt_target(lams):
    eps = 1e-4
    return np.sqrt(np.maximum(lams - lambda0, eps))

# =========================================================
# LOSS FUNCTION: SQRT BRANCH POINT
# =========================================================
def loss_sqrt_pole(x,
                   w_fit=1.0,
                   w_pole=3.0,
                   w_amp=0.2):

    params = decode(x)
    phis = compute_phase(params)

    # Phase offset irrelevant
    idx0 = np.argmin(np.abs(lambdas - lambda0))
    phis -= phis[idx0]

    # ---- 1. Fit sqrt law AWAY from pole ----
    mask = lambdas > lambda0 + 0.003
    tgt = sqrt_target(lambdas[mask])

    A = np.dot(phis[mask], tgt) / np.dot(tgt, tgt)
    fit_error = np.mean((phis[mask] - A * tgt)**2)

    # ---- 2. Enforce divergence at pole ----
    dphi = np.gradient(phis, lambdas)
    pole_strength = -np.abs(dphi[idx0])

    # ---- 3. Penalize unphysical amplitudes ----
    amp_penalty = 0.0
    for lam in lambdas:
        r00 = solver_system(1/lam, *params)
        amp_penalty += max(0, np.abs(r00) - 1.2)**2

    loss = (
        w_fit * fit_error
        + w_pole * pole_strength
        + w_amp * amp_penalty
    )

    print(
        f"fit={fit_error:.2e}, "
        f"|dφ/dλ|@λ0={abs(dphi[idx0]):.2e}, "
        f"LOSS={loss:.2e}"
    )

    return loss

# =========================================================
# CMA INITIALIZATION (FROM YOUR LINEAR OPTIMUM)
# =========================================================
x0_phys = np.array([
    0.2720562713701102,
    0.23374715620688444,
    0.5741390216475506,
    0.7098414822089394
])

x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
sigma0 = 0.12

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "popsize": 10,
        "maxiter": 60,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_sqrt_pole(x) for x in xs]
    es.tell(xs, losses)

best_x = es.result.xbest
best_params = decode(best_x)

print("\n===== SQRT-POLE DESIGN =====")
print("r     =", best_params[0])
print("h     =", best_params[1])
print("hsio2 =", best_params[2])
print("a     =", best_params[3])

# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(best_params)
dphi = np.gradient(phis, lambdas)

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")
plt.plot(lambdas, sqrt_target(lambdas) * np.max(phis), '--', label="√(λ−λ₀) ref")
plt.axvline(lambda0, color='r', linestyle=':', label="λ₀")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Square-root phase dispersion with pole")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, dphi, 'o-')
plt.axvline(lambda0, color='r', linestyle=':')
plt.xlabel("Wavelength (µm)")
plt.ylabel("dφ/dλ (rad/µm)")
plt.title("Phase derivative (pole behavior)")
plt.grid(True)
plt.tight_layout()
plt.show()
