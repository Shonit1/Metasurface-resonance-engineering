import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# Wavelength range (µm) — narrowband for Huygens physics
# =========================================================
lambdas = np.linspace(1.5, 1.55, 10)

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
# RCWA solver — TWO patterned silicon layers (Huygens)
# =========================================================
def solver_system(f, pattern1, pattern2, h1, h2, hsio2, a):

    es = epsilon_lambda(1 / f)
    L1 = [a, 0]
    L2 = [0, a]

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    # Superstrate
    obj.Add_LayerUniform(0.1, eair)

    # Grid layer 1
    obj.Add_LayerGrid(h1, Nx, Ny)

    # Spacer
    obj.Add_LayerUniform(hsio2, esio2)

    # Grid layer 2
    obj.Add_LayerGrid(h2, Nx, Ny)

    # Substrate (air for now – true Huygens config)
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    # Build ε-grids
    ep1 = get_epgrid_3x3(pattern1, es, a).flatten()
    ep2 = get_epgrid_3x3(pattern2, es, a).flatten()

    # 🔑 CRITICAL FIX: concatenate grids
    obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    Ri,Ti = obj.RT_Solve(normalize = 1,byorder=1)



    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0],Ri,Ti
    else:
        return bi[k0 + nV] / ai[k0 + nV],Ri,Ti



# =========================================================
# Phase computation
# =========================================================
def compute_phase(params):

    pattern1 = params[0:9]
    pattern2 = params[9:18]
    h1, h2, hsio2, a = params[18:]

    phis = []
    for lam in lambdas:
        r00 = solver_system(
            1 / lam,
            pattern1, pattern2,
            h1, h2, hsio2, a
        )
        phis.append(np.angle(r00))

    return np.unwrap(np.array(phis))


# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0, 1]] * 18 +     # 2 × (3×3) pixel patterns
    [
        [0.05, 0.35],  # h1
        [0.05, 0.35],  # h2
        [0.05, 0.40],  # spacer
        [0.1, 0.7]   # lattice constant
    ]
)

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# LOSS FUNCTION — Huygens condition
# =========================================================
def loss_function(x, gamma=20.0):

    params = decode(x)
    phis = compute_phase(params)

    A, B = np.polyfit(lambdas, phis, 1)
    loss = A**2 

    print(f"A={A:.3e}, LOSS={loss:.3e}")
    return loss


# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array(
    [0.5] * 18 +       # two half-filled patterns
    [0.15, 0.15, 0.20, 0.60]
)

x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

es = cma.CMAEvolutionStrategy(
    x0,
    0.25,
    {
        "popsize": 12,
        "maxiter": 40,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])

best = decode(es.result.xbest)

# =========================================================
# RESULTS
# =========================================================
print("\n===== HUYGGENS METASURFACE RESULT =====")

print("Pattern layer 1:")
print((best[:9] > 0.5).reshape(3, 3).astype(int))

print("Pattern layer 2:")
print((best[9:18] > 0.5).reshape(3, 3).astype(int))

print("h1 =", best[18])
print("h2 =", best[19])
print("spacer =", best[20])
print("a =", best[21])

phis = compute_phase(best)

plt.figure()
plt.plot(lambdas, phis, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Two-layer Huygens metasurface phase")
plt.grid(True)
plt.show()
