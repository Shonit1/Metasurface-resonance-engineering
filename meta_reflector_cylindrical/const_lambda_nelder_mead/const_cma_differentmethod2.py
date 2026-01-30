import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# GLOBAL SETTINGS
# =========================================================
WINDOW = 0.05     # 50 nm
N_LAM = 15

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
    x, y = np.meshgrid(x0, y0, indexing="ij")
    x -= a / 2
    y -= a / 2
    mask = (x**2 + y**2) < r**2
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = et
    return ep


# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1, L2 = [a, 0], [0, a]
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

    obj.GridLayer_geteps(get_epgrids_cylinder(r, es, a).flatten())

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    return (
        bi[k0] / ai[k0]
        if abs(ai[k0]) > abs(ai[k0 + nV])
        else bi[k0 + nV] / ai[k0 + nV]
    )


# =========================================================
# PHASE AT SINGLE WAVELENGTH (NO UNWRAP)
# =========================================================
def phase_at_lambda(params, lam):
    r, h, hsio2, a = params
    return np.angle(solver_system(1 / lam, r, h, hsio2, a))


# =========================================================
# GEOMETRY PARAMETERIZATION (a >= 2r ENFORCED)
# =========================================================
geom_bounds = np.array([
    [0.05, 0.45],   # r
    [0.05, 0.40],   # h
    [0.05, 0.60],   # hsio2
    [0.00, 0.40],   # delta_a = a - 2r
])

def decode_geom(x):
    x = np.clip(x, 0.0, 1.0)
    raw = geom_bounds[:, 0] + x * (geom_bounds[:, 1] - geom_bounds[:, 0])
    r, h, hsio2, delta_a = raw
    a = 2.0 * r + delta_a
    return np.array([r, h, hsio2, a])


# =========================================================
# ==================== PHASE 1 ============================
# Two-point slope minimization
# =========================================================
lambda_1 = 1.4
lambda_2 = 1.6

def loss_phase1(x):
    params = decode_geom(x)

    phi1 = phase_at_lambda(params, lambda_1)
    phi2 = phase_at_lambda(params, lambda_2)

    dphi = np.angle(np.exp(1j * (phi2 - phi1)))
    slope = dphi / (lambda_2 - lambda_1)

    print(f"[P1] slope = {slope:.3e}")
    return slope**2


print("\n=== PHASE 1: TWO-POINT SLOPE MINIMIZATION ===")

es1 = cma.CMAEvolutionStrategy(
    np.random.rand(4), 0.3,
    {"popsize": 10, "maxiter": 30}
)

while not es1.stop():
    xs = es1.ask()
    es1.tell(xs, [loss_phase1(x) for x in xs])

best_geom = decode_geom(es1.result.xbest)
print("\nBest geometry after Phase 1:", best_geom)


# =========================================================
# ==================== PHASE 2 ============================
# Optimize wavelength window center
# =========================================================
lambda_bounds = np.array([1.251, 1.949])

def decode_lambda(x):
    x = np.clip(x[0], 0.0, 1.0)
    return lambda_bounds[0] + x * (lambda_bounds[1] - lambda_bounds[0])


def slope_in_window(lambda0):
    lambdas = lambda0 + np.linspace(-WINDOW/2, WINDOW/2, N_LAM)
    phis = [phase_at_lambda(best_geom, lam) for lam in lambdas]
    phis = np.unwrap(np.array(phis))
    A, _ = np.polyfit(lambdas, phis, 1)
    return A


def loss_phase2(x):
    lambda0 = decode_lambda(x)
    A = slope_in_window(lambda0)
    print(f"[P2] λ0 = {lambda0:.4f}, A = {A:.3e}")
    return A**2


print("\n=== PHASE 2: WINDOW SEARCH ===")

es2 = cma.CMAEvolutionStrategy(
    [0.5], 0.2,
    {"popsize": 8, "maxiter": 25}
)

while not es2.stop():
    xs = es2.ask()
    es2.tell(xs, [loss_phase2(x) for x in xs])

lambda0_opt = decode_lambda(es2.result.xbest)
print("\nBest window center λ0:", lambda0_opt)


# =========================================================
# ==================== PHASE 3 ============================
# Geometry refinement in best window
# =========================================================
def loss_phase3(x):
    params = decode_geom(x)
    lambdas = lambda0_opt + np.linspace(-WINDOW/2, WINDOW/2, N_LAM)
    phis = [phase_at_lambda(params, lam) for lam in lambdas]
    phis = np.unwrap(np.array(phis))

    A, B = np.polyfit(lambdas, phis, 1)
    rms = np.std(phis - (A * lambdas + B))

    print(f"[P3] A = {A:.3e}, RMS = {rms:.3e}")
    return 200 * A**2 + 10 * rms


print("\n=== PHASE 3: GEOMETRY REFINEMENT ===")

es3 = cma.CMAEvolutionStrategy(
    np.random.rand(4), 0.15,
    {"popsize": 8, "maxiter": 40}
)

while not es3.stop():
    xs = es3.ask()
    es3.tell(xs, [loss_phase3(x) for x in xs])

best_final = decode_geom(es3.result.xbest)
print("\nFINAL GEOMETRY:", best_final)


# =========================================================
# FINAL DIAGNOSTIC PLOT
# =========================================================
lambdas = lambda0_opt + np.linspace(-WINDOW/2, WINDOW/2, N_LAM)
phis = [phase_at_lambda(best_final, lam) for lam in lambdas]
phis = np.unwrap(np.array(phis))
A, B = np.polyfit(lambdas, phis, 1)

plt.plot(lambdas, phis, 'o-', label="Phase")
plt.plot(lambdas, A*lambdas + B, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title(f"Final slope A = {A:.3e}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
