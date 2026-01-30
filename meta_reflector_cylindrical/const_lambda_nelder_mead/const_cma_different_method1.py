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
lambdas = np.linspace(1.575, 1.625, 25)

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
# Phase computation
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
    [0.3, 0.5],   # r
    [0.15, 0.25],   # h
    [0.25, 0.35],   # hsio2
    [0.8, 1]    # a
])


# =========================================================
# CMA BOUND HANDLER
# =========================================================
def decode(x):
    x = np.clip(x, 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# LOSS FUNCTION → FORCE A → 0
# =========================================================
def loss_function(x, alpha=10.0, beta=100.0, gamma=200.0):

    params = decode(x)
    phis = compute_phase(params)

    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    d2phi = np.gradient(np.gradient(phis, lambdas), lambdas)
    curvature = np.mean(d2phi**2)

    # Key change: penalize slope
    loss = gamma * A**2 + alpha * rms + beta * curvature

    print(
        f"A={A:.4e}, RMS={rms:.3e}, CURV={curvature:.3e}, LOSS={loss:.3e}"
    )
    return loss


# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array([0.4, 0.2, 0.3, 0.9])
x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

sigma0 = 0.15

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "popsize": 8,
        "maxiter": 40,
        "verb_disp": 1
    }
)


# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    solutions = es.ask()
    losses = [loss_function(x) for x in solutions]
    es.tell(solutions, losses)

best_x = es.result.xbest
best_params = decode(best_x)

r_opt, h_opt, hsio2_opt, a_opt = best_params

print("\n===== CMA OPTIMUM =====")
print("r     =", r_opt)
print("h     =", h_opt)
print("hsio2 =", hsio2_opt)
print("a     =", a_opt)


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(best_params)
A_opt, B_opt = np.polyfit(lambdas, phis, 1)
phi_fit = A_opt * lambdas + B_opt

print("\n===== FINAL LINEAR FIT =====")
print("A (rad/µm) =", A_opt)
print("B (rad)    =", B_opt)
print("RMS error  =", np.std(phis - phi_fit))


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")
plt.plot(lambdas, phi_fit, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("CMA-optimized flat phase (A ≈ 0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis - phi_fit, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Deviation from flatness")
plt.grid(True)
plt.tight_layout()
plt.show()
