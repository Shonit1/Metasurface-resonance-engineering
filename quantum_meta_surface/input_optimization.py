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
lambdas = np.linspace(1.5, 1.505, 10)

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
    pattern = np.array(pattern).reshape(3, 3)
    pattern = (pattern > 0.5).astype(int)

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

    es = epsilon_lambda(1 / f)
    L1 = [a, 0]
    L2 = [0, a]

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

    obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


def compute_phase(params):
    pattern = params[:9]
    h, hsio2, a = params[9:]
    return np.unwrap([
        np.angle(solver_system(1 / lam, pattern, h, hsio2, a))
        for lam in lambdas
    ])


# =========================================================
# BOUNDS
# =========================================================
bounds = np.array(
    [[0, 1]] * 9 +
    [
        [0.05, 0.40],
        [0.05, 0.60],
        [0.2, 0.80]
    ]
)

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# PHASE 1 — SLOPE MINIMIZATION
# =========================================================
candidates = []

def loss_phase1(x):
    params = decode(x)
    phis = compute_phase(params)
    A, _ = np.polyfit(lambdas, phis, 1)
    loss = A**2
    candidates.append((loss, params.copy()))
    print(f"[P1] A={A:.3e}")
    return loss


x0 = np.array([0.5]*9 + [0.12, 0.30, 0.60])
x0 = (x0 - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

es = cma.CMAEvolutionStrategy(x0, 0.25, {"popsize": 10, "maxiter": 40})

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_phase1(x) for x in xs])

# Select top 10 candidates
candidates = sorted(candidates, key=lambda x: x[0])[:10]


# =========================================================
# PHASE 2 — HUMAN SELECTION
# =========================================================
plt.figure(figsize=(7, 5))
for i, (_, p) in enumerate(candidates):
    phis = compute_phase(p)
    plt.plot(lambdas, phis, label=f"{i}")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Top 10 near-zero slope candidates")
plt.legend()
plt.grid(True)
plt.show()

choice = int(input("Select candidate index (0–9): "))
start_params = candidates[choice][1]


# =========================================================
# PHASE 3 — RMS + CURVATURE REFINEMENT
# =========================================================
def loss_phase3(x, alpha=10, beta=100, gamma=5):
    params = decode(x)
    phis = compute_phase(params)
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B
    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    d2phi = np.gradient(np.gradient(phis, lambdas), lambdas)
    curvature = np.mean(d2phi**2)
    loss = gamma*A**2 + alpha*rms + beta*curvature
    print(f"[P3] A={A:.2e}, RMS={rms:.2e}, CURV={curvature:.2e}")
    return loss


x0 = (start_params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
es = cma.CMAEvolutionStrategy(x0, 0.1, {"popsize": 8, "maxiter": 30})

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_phase3(x) for x in xs])

best = decode(es.result.xbest)


# =========================================================
# FINAL RESULT
# =========================================================
print("\n===== FINAL DESIGN =====")
print("Pattern:")
print((best[:9] > 0.5).reshape(3, 3).astype(int))
print("h =", best[9])
print("hsio2 =", best[10])
print("a =", best[11])

phis = compute_phase(best)
plt.figure()
plt.plot(lambdas, phis, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Final optimized constant phase")
plt.grid(True)
plt.show()
