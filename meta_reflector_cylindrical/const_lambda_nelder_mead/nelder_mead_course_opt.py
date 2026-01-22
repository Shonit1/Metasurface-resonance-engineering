import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
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
# Cylinder ε-grid (square lattice)
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
        a00, b00 = ai[k0], bi[k0]
    else:
        a00, b00 = ai[k0 + nV], bi[k0 + nV]

    return b00 / a00


# =========================================================
# Unwrapped phase spectrum
# =========================================================
def compute_phase_spectrum(r, h, hsio2, a):

    phis = []
    for lam in lambdas:
        f = 1 / lam
        r00 = solver_system(f, r, h, hsio2, a)
        phis.append(np.angle(r00))

    return np.unwrap(np.array(phis))


# =========================================================
# Automatic L0 fitting
# =========================================================
def fit_L0(phis):
    """
    Least-squares optimal L0 such that
    phi ≈ 2π L0 / λ
    """
    x = 1 / lambdas
    A = np.vstack([2*np.pi*x]).T
    L0_opt, *_ = np.linalg.lstsq(A, phis, rcond=None)
    return L0_opt[0]


# =========================================================
# OBJECTIVE:
# constant residual phase after optimal L0 subtraction
# =========================================================
def objective(params):
    r, h, hsio2, a = params

    if not (0.01 <= a <= 1.2):
        return 1e3
    if not (0.01 <= r <= 0.45 * a):
        return 1e3
    if not (0.01 <= h <= 1.2):
        return 1e3
    if not (0.05 <= hsio2 <= 1.5):
        return 1e3

    phis = compute_phase_spectrum(r, h, hsio2, a)

    L0_opt = fit_L0(phis)
    phi_ref = 2 * np.pi * L0_opt / lambdas

    phi_res = phis - phi_ref
    C = np.mean(phi_res)

    return np.sqrt(np.mean((phi_res - C)**2))


# =========================================================
# Final optimization
# =========================================================
import cma

# =========================================================
# CMA-ES GLOBAL OPTIMIZATION
# =========================================================

# Initial guess (same as before)
x0 = np.array([0.23, 0.12, 0.3, 0.6])

# Initial step size (important!)
sigma0 = 0.15

# Parameter bounds: [lower, upper]
bounds = [
    [0.05, 0.01, 0.05, 0.4],   # lower bounds
    [0.45, 1.2, 1.5, 1.2]     # upper bounds
]

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": bounds,
        "popsize": 16,        # good tradeoff for RCWA
        "verb_disp": 1,
        "maxiter": 40         # keep small (expensive objective!)
    }
)

while not es.stop():
    solutions = es.ask()
    losses = []

    for x in solutions:
        loss = objective(x)
        losses.append(loss)

    es.tell(solutions, losses)

    es.disp()

# Best solution from CMA
x_cma = es.result.xbest
print("\nCMA-ES result:")
print("r =", x_cma[0])
print("h =", x_cma[1])
print("hsio2 =", x_cma[2])
print("a =", x_cma[3])
print("CMA loss =", es.result.fbest)


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase_spectrum(x_cma[0], x_cma[1], x_cma[2], x_cma[3])

L0_opt = fit_L0(phis)
phi_ref = 2 * np.pi * L0_opt / lambdas
phi_res = phis - phi_ref
C = np.mean(phi_res)

print("Optimal L0 =", L0_opt)
print("Residual RMS (rad) =", np.std(phi_res))

# =========================================================
# PLOTS
# =========================================================
plt.figure()
plt.plot(lambdas, phis, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("Absolute reflection phase")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambdas, phi_res, 'o-', label="Residual phase")
plt.plot(lambdas, C*np.ones_like(lambdas), '--', label="Constant fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Reference-subtracted phase (target = constant)")
plt.legend()
plt.grid(True)
plt.show()
