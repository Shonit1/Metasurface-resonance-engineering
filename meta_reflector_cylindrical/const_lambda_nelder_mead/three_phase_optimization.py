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
N_LAM = 25
WINDOW = 0.10   # µm

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

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    return (
        bi[k0] / ai[k0]
        if abs(ai[k0]) > abs(ai[k0 + nV])
        else bi[k0 + nV] / ai[k0 + nV]
    )


# =========================================================
# PHASE COMPUTATION
# =========================================================
def compute_phase(params, lambdas):
    r, h, hsio2, a = params
    phis = []
    for lam in lambdas:
        phis.append(np.angle(solver_system(1 / lam, r, h, hsio2, a)))
    return np.unwrap(np.array(phis))


# =========================================================
# BASE BOUNDS (NOTE: delta_a, not a)
# =========================================================
base_bounds = np.array([
    [0.05, 0.45],   # r
    [0.05, 0.40],   # h
    [0.05, 0.60],   # hsio2
    [0.00, 0.40],   # delta_a = a - 2r
])


# =========================================================
# DECODE (ENFORCES a >= 2r)
# =========================================================
def decode(x, bounds):
    x = np.clip(x, 0.0, 1.0)
    raw = bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

    r, h, hsio2, delta_a = raw
    a = 2.0 * r + delta_a   # CONSTRAINT ENFORCED

    return np.array([r, h, hsio2, a])


# =========================================================
# ---------------- PHASE 1 ----------------
# Candidate discovery (slope stability)
# =========================================================
def loss_phase1(x):
    params = decode(x, base_bounds)
    lambdas = np.linspace(1.4, 1.6, N_LAM)
    phis = compute_phase(params, lambdas)

    A_list = []
    for i in range(len(lambdas) - 4):
        A, _ = np.polyfit(lambdas[i:i+5], phis[i:i+5], 1)
        A_list.append(A)

    A_list = np.array(A_list)
    A_var = np.var(A_list)
    A_mean = np.mean(A_list)

    d2phi = np.gradient(np.gradient(phis, lambdas), lambdas)
    curvature = np.mean(d2phi**2)

    loss = 10*A_var + 50*curvature + 0.2*A_mean**2
    print(f"[P1] Ā={A_mean:.3f}, var={A_var:.2e}, loss={loss:.3e}")
    return loss


print("\n=== PHASE 1: FINDING STABLE CANDIDATES ===")

es1 = cma.CMAEvolutionStrategy(np.random.rand(4), 0.3,
                               {"popsize": 10, "maxiter": 30})

while not es1.stop():
    xs = es1.ask()
    es1.tell(xs, [loss_phase1(x) for x in xs])

best_p1 = decode(es1.result.xbest, base_bounds)
print("\nBest Phase-1 params:", best_p1)


# =========================================================
# ---------------- PHASE 2 ----------------
# Local refinement (A → 0)
# =========================================================
print("\n=== PHASE 2: LOCAL FLATTENING ===")

r0, h0, hs0, a0 = best_p1
ref_bounds = np.array([
    [r0 - 0.03, r0 + 0.03],
    [h0 - 0.03, h0 + 0.03],
    [hs0 - 0.04, hs0 + 0.04],
    [0.00, 0.15],   # delta_a (still ≥ 0)
])

def loss_phase2(x):
    params = decode(x, ref_bounds)
    lambdas = np.linspace(1.45, 1.55, N_LAM)
    phis = compute_phase(params, lambdas)

    A, B = np.polyfit(lambdas, phis, 1)
    rms = np.std(phis - (A*lambdas + B))
    d2phi = np.gradient(np.gradient(phis, lambdas), lambdas)
    curvature = np.mean(d2phi**2)

    loss = 300*A**2 + 20*rms + 100*curvature
    print(f"[P2] A={A:.3e}, loss={loss:.3e}")
    return loss


es2 = cma.CMAEvolutionStrategy(np.full(4, 0.5), 0.2,
                               {"popsize": 8, "maxiter": 40})

while not es2.stop():
    xs = es2.ask()
    es2.tell(xs, [loss_phase2(x) for x in xs])

best_p2 = decode(es2.result.xbest, ref_bounds)
print("\nBest Phase-2 params:", best_p2)


# =========================================================
# ---------------- PHASE 3 ----------------
# Optimize window center λ0
# =========================================================
print("\n=== PHASE 3: WINDOW CENTERING ===")

def loss_phase3(x):
    lambda0 = 1.45 + 0.10 * x[0]
    lambdas = lambda0 + np.linspace(-WINDOW/2, WINDOW/2, N_LAM)
    phis = compute_phase(best_p2, lambdas)
    A, _ = np.polyfit(lambdas, phis, 1)
    print(f"[P3] λ0={lambda0:.4f}, A={A:.3e}")
    return A**2


es3 = cma.CMAEvolutionStrategy([0.5], 0.2,
                               {"popsize": 6, "maxiter": 20})

while not es3.stop():
    xs = es3.ask()
    es3.tell(xs, [loss_phase3(x) for x in xs])

lambda0_opt = 1.45 + 0.10 * es3.result.xbest[0]
print("\nOptimal λ0 =", lambda0_opt)


# =========================================================
# FINAL DIAGNOSTIC
# =========================================================
lambdas = lambda0_opt + np.linspace(-WINDOW/2, WINDOW/2, N_LAM)
phis = compute_phase(best_p2, lambdas)
A, B = np.polyfit(lambdas, phis, 1)

plt.plot(lambdas, phis, 'o-', label="Phase")
plt.plot(lambdas, A*lambdas + B, '--', label="Linear fit")
plt.title(f"Final slope A = {A:.3e}")
plt.xlabel("λ (µm)")
plt.ylabel("Phase (rad)")
plt.grid()
plt.legend()
plt.show()
