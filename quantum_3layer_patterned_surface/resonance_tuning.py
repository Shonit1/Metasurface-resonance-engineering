import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# WAVELENGTH GRID
# =========================================================
lambdas = np.linspace(1.490, 1.55, 80)
lambda0 = 1.516   # target branch point


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
# 3×3 PIXEL GRID
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    pattern = np.array(pattern).reshape(3, 3)
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair

    dx = a / 3
    dy = a / 3

    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i, j] > 0.5:
                mask = (
                    (X >= i * dx) & (X < (i + 1) * dx) &
                    (Y >= j * dy) & (Y < (j + 1) * dy)
                )
                epgrid[mask] = et

    return epgrid


# =========================================================
# RCWA SOLVER (TWO GRID LAYERS)
# =========================================================
def solver_system(f, pattern1, pattern2, h1, h2, hsio2, a):

    es = epsilon_lambda(1 / f)

    obj = grcwa.obj(
        nG,
        [a, 0],
        [0, a],
        f,
        theta,
        phi,
        verbose=0
    )

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h1, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)
    obj.Add_LayerGrid(h2, Nx, Ny)
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    ep1 = get_epgrid_3x3(pattern1, es, a).flatten()
    ep2 = get_epgrid_3x3(pattern2, es, a).flatten()

    obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    k0 = np.where(
        (obj.G[:, 0] == 0) & (obj.G[:, 1] == 0)
    )[0][0]

    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


# =========================================================
# SQRT TARGET FUNCTION
# =========================================================
def sqrt_target(lams):
    eps = 1e-5
    return np.sqrt(np.maximum(lams - lambda0, eps))


# =========================================================
# PHASE COMPUTATION
# =========================================================
def compute_phase(params):

    p1 = params[0:9]
    p2 = params[9:18]
    h1, h2, hsio2, a = params[18:]

    phis = []

    for lam in lambdas:
        r = solver_system(1 / lam, p1, p2, h1, h2, hsio2, a)
        phis.append(np.angle(r))

    phis = np.unwrap(np.array(phis))

    # remove global phase
    idx0 = np.argmin(np.abs(lambdas - lambda0))
    phis -= phis[idx0]

    return phis


# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0, 1]] * 18 +       # pixel DOFs
    [
        [0.05, 0.35],    # h1
        [0.05, 0.35],    # h2
        [0.03, 0.20],    # spacer
        [0.60, 0.85]     # lattice constant
    ]
)

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# LOSS FUNCTION — STRICT SQRT POLE
# =========================================================
def loss_sqrt_pole(x,
                   w_fit=0,
                   w_pole=3.0,
                   w_reg=0):

    params = decode(x)
    phis = compute_phase(params)

    # ---- 1. √-fit AWAY from pole
    mask = lambdas > lambda0 + 0.01
    tgt = sqrt_target(lambdas[mask])

    A = np.dot(phis[mask], tgt) / np.dot(tgt, tgt)
    fit_error = np.mean((phis[mask] - A * tgt)**2)

    # ---- 2. Divergent derivative at pole
    dphi = np.gradient(phis, lambdas)
    idx0 = np.argmin(np.abs(lambdas - lambda0))
    pole_strength = -abs(dphi[idx0])

    # ---- 3. Regularization (avoid crazy oscillations)
    reg = np.mean(np.diff(dphi)**2)

    loss = (
        w_fit * fit_error
        + w_pole * pole_strength
        + w_reg * reg
    )

    print(
        f"fit={fit_error:.2e}, "
        f"|dφ/dλ|@λ0={abs(dphi[idx0]):.2e}, "
        f"LOSS={loss:.2e}"
    )

    return loss


# =========================================================
# CMA INITIALIZATION (START FROM YOUR GEOMETRY)
# =========================================================
x0_phys = np.array(
    [
        0,0,0, 1,1,1, 0,0,1,     # pattern1
        1,1,1, 1,1,1, 0,0,0,     # pattern2
        0.2437,                  # h1
        0.35,                    # h2
        0.05,                    # spacer
        0.8                      # a
    ]
)

x0 = (x0_phys - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

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
    es.tell(xs, [loss_sqrt_pole(x) for x in xs])

best = decode(es.result.xbest)


# =========================================================
# RESULTS
# =========================================================
print("\n===== OPTIMIZED SQRT-POLE METASURFACE =====")

print("Pattern layer 1:")
print((best[:9] > 0.5).reshape(3,3).astype(int))

print("Pattern layer 2:")
print((best[9:18] > 0.5).reshape(3,3).astype(int))

print("h1 =", best[18])
print("h2 =", best[19])
print("spacer =", best[20])
print("a =", best[21])


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(best)
dphi = np.gradient(phis, lambdas)

plt.figure()
plt.plot(lambdas, phis, 'k', lw=2)
plt.axvline(lambda0, ls=':', color='gray')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Optimized √-branch-point phase")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(lambdas, dphi, 'r')
plt.axvline(lambda0, ls=':', color='gray')
plt.xlabel("Wavelength (µm)")
plt.ylabel(r"$d\phi/d\lambda$")
plt.title("Pole-like derivative divergence")
plt.grid(True)
plt.show()
