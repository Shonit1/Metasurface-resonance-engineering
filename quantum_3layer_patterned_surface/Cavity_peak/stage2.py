import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# STAGE-2 WINDOW (LOCK + START SQUEEZING)
# =========================================================
lambdas_res = np.linspace(1.4997, 1.5003, 41)

# =========================================================
# 🔴 MANUALLY PASTED STAGE-1 GEOMETRY
# =========================================================
P1_INIT = np.array([
    1,1,1,
    0,1,1,
    1,0,0
])

P2_INIT = np.array([
    1,0,0,
    0,0,1,
    0,0,0
])

h1_INIT = 0.295713
h2_INIT = 0.093016
hs_INIT = 0.217735
hsio2_INIT = 0.157112
a_INIT = 0.171713


# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0].values,
            data.iloc[:,1].values,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    n = 3.7 if wavelength < 1.0 else _cache["li"](wavelength)
    return n**2


# =========================================================
# GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = pattern.reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i, j]:
                ep[
                    (X >= i*dx) & (X < (i+1)*dx) &
                    (Y >= j*dy) & (Y < (j+1)*dy)
                ] = eps
    return ep


# =========================================================
# RCWA SOLVER (ONE WAVELENGTH)
# =========================================================
def solver_system(f, h1, h2, hs, hsio2, a):
    try:
        eps_si = epsilon_lambda(1/f)
        obj = grcwa.obj(nG, [a,0], [0,a], f, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1_INIT, eps_si, a).flatten()
        ep2 = get_epgrid_3x3(P2_INIT, eps_si, a).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0)

        _, T = obj.RT_Solve(normalize=1, byorder=1)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]

        return T[k0]

    except Exception:
        return None


# =========================================================
# SPECTRUM
# =========================================================
def compute_spectrum(h1, h2, hs, hsio2, a):
    T = []
    for lam in lambdas_res:
        val = solver_system(1/lam, h1, h2, hs, hsio2, a)
        if val is None or np.isnan(val):
            return None
        T.append(val)
    return np.array(T)


def center_second_derivative(y, dx):
    c = len(y) // 2
    return (y[c+1] - 2*y[c] + y[c-1]) / (dx*dx)


# =========================================================
# LOSS FUNCTION (CURVATURE + SLOPE-RATIO)
# =========================================================
def loss_function(x):

    h1, h2, hs, hsio2, a = x
    T = compute_spectrum(h1, h2, hs, hsio2, a)

    if T is None or len(T) < 3:
        return 1e9

    dx = lambdas_res[1] - lambdas_res[0]
    c = len(T) // 2

    # --- curvature (resonance existence) ---
    d2T = center_second_derivative(T, dx)
    curvature_penalty = max(0, d2T)

    # --- edge rejection ---
    edge_penalty = np.sum(np.maximum(0, T[[0, -1]] - T[c]))

    # --- slope-ratio proxy (linewidth) ---
    center_slope = abs(T[c+1] - T[c-1]) / (2 * dx)
    edge_slope = abs(T[-1] - T[0]) / (lambdas_res[-1] - lambdas_res[0])
    slope_ratio = edge_slope / (center_slope + 1e-12)

    # --- reward strong resonance ---
    transmission_reward = max(T[c], 0)

    loss = (
        1e4 * curvature_penalty +
        1e3 * edge_penalty +
        2e3 * slope_ratio -    # ← THIS drives narrowing
        1e3 * transmission_reward
    )

    print(
        f"Tc={T[c]:.3f}, "
        f"d2T={d2T:.2e}, "
        f"slope_ratio={slope_ratio:.2e}, "
        f"LOSS={loss:.2e}"
    )

    return loss


# =========================================================
# CMA (LOCAL, STABLE)
# =========================================================
x0 = np.array([h1_INIT, h2_INIT, hs_INIT, hsio2_INIT, a_INIT])

bounds = [
    [0.10, 0.35],
    [0.05, 0.35],
    [0.20, 0.28],
    [0.05, 0.40],
    [0.10, 0.70],
]

opts = {
    "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
    "popsize": 8,
    "maxiter": 50,
}

es = cma.CMAEvolutionStrategy(x0, 0.03, opts)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])


# =========================================================
# FINAL RESULT
# =========================================================
best = es.result.xbest
print("\nBEST PARAMETERS (STAGE-2, SLOPE-RATIO):")
print(f"h1={best[0]:.6f}")
print(f"h2={best[1]:.6f}")
print(f"hs={best[2]:.6f}")
print(f"hsio2={best[3]:.6f}")
print(f"a={best[4]:.6f}")

T = compute_spectrum(*best)
plt.plot(lambdas_res, T, "-o")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmission")
plt.title("Stage-2 Resonance (Curvature + Slope-Ratio)")
plt.grid(True)
plt.show()
