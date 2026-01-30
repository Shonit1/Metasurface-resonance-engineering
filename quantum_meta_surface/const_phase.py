import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# Wavelength range (µm)
# =========================================================
lambdas = np.linspace(1.5, 1.55, 25)

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
# 3×3 pixelated ε-grid
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    """
    pattern : length-9 array, values in [0,1]
    """
    pattern = np.array(pattern).reshape(3, 3)
    pattern = (pattern > 0.5).astype(float)

    epgrid = np.ones((Nx, Ny), dtype=complex) * eair

    dx = a / 3
    dy = a / 3

    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i, j] == 1:
                mask = (
                    (X >= i * dx) & (X < (i + 1) * dx) &
                    (Y >= j * dy) & (Y < (j + 1) * dy)
                )
                epgrid[mask] = et

    return epgrid


# =========================================================
# RCWA solver
# =========================================================
def solver_system(f, pattern, h, hsio2, a):

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

    epgrid = get_epgrid_3x3(pattern, es, a).flatten()
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
# Phase spectrum
# =========================================================
def compute_phase_spectrum(params):
    pattern = params[:9]
    h, hsio2, a = params[9:]

    phis = []
    for lam in lambdas:
        r00 = solver_system(1 / lam, pattern, h, hsio2, a)
        phis.append(np.angle(r00))

    return np.unwrap(np.array(phis))


# =========================================================
# Fit effective optical length L0
# =========================================================
def fit_L0(phis):
    x = 1 / lambdas
    A = np.vstack([2 * np.pi * x]).T
    L0, *_ = np.linalg.lstsq(A, phis, rcond=None)
    return L0[0]


# =========================================================
# CMA bounds
# =========================================================
bounds = np.array(
    [[0, 1]] * 9 +      # 9 pixel variables
    [
        [0.01, 0.6],   # h
        [0.1, 1.0],    # hsio2
        [0.2, 0.8]     # lattice constant a
    ]
)

def decode(x):
    x = np.clip(x, 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# CMA loss function
# =========================================================
def loss_cma(x):

    params = decode(x)
    phis = compute_phase_spectrum(params)

    L0 = fit_L0(phis)
    phi_ref = 2 * np.pi * L0 / lambdas
    phi_res = phis - phi_ref

    C = np.mean(phi_res)
    rms = np.sqrt(np.mean((phi_res - C)**2))

    print(f"L0={L0:.4f}, RMS={rms:.3e}")
    return rms


# =========================================================
# CMA setup
# =========================================================
x0_phys = np.array(
    [0.5] * 9 +        # initial half-filled pattern
    [0.12, 0.2, 0.5]
)

sigma0 = 0.25
x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "popsize": 10,
        "maxiter": 50,
        "verb_disp": 1
    }
)

while not es.stop():
    xs = es.ask()
    losses = [loss_cma(x) for x in xs]
    es.tell(xs, losses)
    es.disp()

best_params = decode(es.result.xbest)

print("\n===== CMA OPTIMUM =====")
print("Pattern (3×3):")
print((best_params[:9] > 0.5).reshape(3, 3).astype(int))
print("h     =", best_params[9])
print("hsio2 =", best_params[10])
print("a     =", best_params[11])
print("LOSS  =", es.result.fbest)


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase_spectrum(best_params)
L0 = fit_L0(phis)
phi_ref = 2 * np.pi * L0 / lambdas
phi_res = phis - phi_ref
C = np.mean(phi_res)

print("Optimal L0 =", L0)
print("Residual RMS (rad) =", np.std(phi_res))

plt.figure()
plt.plot(lambdas, phis, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Reflection phase")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambdas, phi_res, 'o-', label="Residual")
plt.plot(lambdas, C * np.ones_like(lambdas), '--', label="Constant")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Constant-phase residual")
plt.legend()
plt.grid(True)
plt.show()
