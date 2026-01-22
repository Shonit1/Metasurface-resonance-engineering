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
lambdas = np.linspace(1.45, 1.50, 10)







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

    obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                               s_amp=0, s_phase=0, order=0)

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
# LOCAL RESONANT BOUNDS (IMPORTANT)
# =========================================================
bounds = np.array([
    [0.25, 0.35],   # r
    [0.06, 0.14],   # h
    [0.35, 0.55],   # hsio2
    [0.60, 0.75]    # a
])


def decode(x):
    x = np.clip(x, 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# STRATEGY 1 LOSS: RESONANCE BROADENING
# =========================================================
def loss_resonant(x, beta=2.0, gamma=0.2):

    params = decode(x)
    phis = compute_phase(params)

    # First derivative (slope)
    dphi = np.gradient(phis, lambdas)

    # Second derivative (curvature)
    d2phi = np.gradient(dphi, lambdas)

    mean_slope = np.mean(np.abs(dphi))
    slope_spread = np.std(dphi)
    curvature = np.mean(d2phi**2)

    loss = (
        -mean_slope
        + beta * slope_spread
        + gamma * curvature
    )

    print(
        f"<|dφ/dλ|>={mean_slope:.2f}, "
        f"std={slope_spread:.2f}, "
        f"curv={curvature:.2e}, "
        f"LOSS={loss:.2f}"
    )

    return loss


# =========================================================
# CMA INITIALIZATION (FROM YOUR OPTIMUM)
# =========================================================
x0_phys = np.array([
    0.2918330181766115,
    0.09149736053023741,
    0.4525771172692921,
    0.667828504940109
])

x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
sigma0 = 0.10

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "popsize": 8,
        "maxiter": 50,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_resonant(x) for x in xs]
    es.tell(xs, losses)

best_x = es.result.xbest
best_params = decode(best_x)

print("\n===== BROADENED RESONANT DESIGN =====")
print("r     =", best_params[0])
print("h     =", best_params[1])
print("hsio2 =", best_params[2])
print("a     =", best_params[3])


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(best_params)
dphi = np.gradient(phis, lambdas)

A, B = np.polyfit(lambdas, phis, 1)
phi_fit = A * lambdas + B

print("\n===== FINAL METRICS =====")
print("Linear-fit slope A (rad/µm):", A)
print("Mean |dφ/dλ|:", np.mean(np.abs(dphi)))
print("Slope std:", np.std(dphi))


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="Phase")
plt.plot(lambdas, phi_fit, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Resonance-broadened phase response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, dphi, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("dφ/dλ (rad/µm)")
plt.title("Distributed resonant dispersion")
plt.grid(True)
plt.tight_layout()
plt.show()
