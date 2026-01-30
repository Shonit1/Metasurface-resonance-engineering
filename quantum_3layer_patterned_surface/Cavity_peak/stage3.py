import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# STAGE-3: ULTRA-HIGH-Q WINDOW
# =========================================================
lambdas_res = np.linspace(1.499, 1.501, 61)
TARGET_LAMBDA = 1.5000

# =========================================================
# 🔒 FROZEN PATTERNS (PASTE FROM STAGE-2)
# =========================================================
P1_FIXED = np.array([
    1,0,1,
    0,1,0,
    1,0,1
])

P2_FIXED = np.array([
    0,1,0,
    1,0,1,
    0,1,0
])

# =========================================================
# MATERIAL MODEL ε(λ)
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

    if wavelength < 1.0:
        n = 3.7
    else:
        n = _cache["li"](wavelength)

    return n**2


# =========================================================
# GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = pattern.reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[
                    (X>=i*dx)&(X<(i+1)*dx)&
                    (Y>=j*dy)&(Y<(j+1)*dy)
                ] = eps
    return ep


# =========================================================
# RCWA SOLVER
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

        ep1 = get_epgrid_3x3(P1_FIXED, eps_si, a).flatten()
        ep2 = get_epgrid_3x3(P2_FIXED, eps_si, a).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0)

        _, _, T = obj.RT_Solve(normalize=1, byorder=1)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]

        return T[k0]

    except Exception:
        return None


# =========================================================
# FWHM ESTIMATOR
# =========================================================
def estimate_fwhm(lambdas, T):
    Tmax = np.max(T)
    if Tmax < 1e-5:
        return None
    half = 0.5 * Tmax
    idx = np.where(T >= half)[0]
    if len(idx) < 2:
        return None
    return lambdas[idx[-1]] - lambdas[idx[0]]


# =========================================================
# STAGE-3 LOSS (LOG-Q)
# =========================================================
def loss_function(x):
    h1, h2, hs, hsio2, a = x

    T = []
    for lam in lambdas_res:
        val = solver_system(1/lam, h1, h2, hs, hsio2, a)
        if val is None:
            return 1e9
        T.append(val)

    T = np.array(T)
    idx = np.argmax(T)
    lambda0 = lambdas_res[idx]
    fwhm = estimate_fwhm(lambdas_res, T)

    if fwhm is None:
        return 1e9

    Q = lambda0 / fwhm

    loss = (
        -np.log10(Q) +
        1e4 * (lambda0 - TARGET_LAMBDA)**2
    )

    print(
        f"λ0={lambda0:.6f}, "
        f"FWHM={fwhm:.2e}, "
        f"Q={Q:.2e}, "
        f"LOSS={loss:.2e}"
    )

    return loss












# =========================================================
# CMA SETUP (VERY TIGHT)
# =========================================================
x0 = np.array([0.20, 0.20, 0.24, 0.15, 0.45])
sigma = 0.01

bounds = [
    [0.10, 0.30],  # h1
    [0.10, 0.30],  # h2
    [0.22, 0.27],  # hs
    [0.05, 0.30],  # hsio2
    [0.30, 0.60],  # a
]

opts = {
    "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
    "popsize": 6,
    "maxiter": 60,
}

es = cma.CMAEvolutionStrategy(x0, sigma, opts)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])


# =========================================================
# FINAL BEST
# =========================================================
best = es.result.xbest
print("\nBEST PARAMETERS:")
print(f"h1={best[0]:.4f}, h2={best[1]:.4f}")
print(f"hs={best[2]:.4f}, hsio2={best[3]:.4f}")
print(f"a={best[4]:.4f}")

T = [solver_system(1/lam, *best) for lam in lambdas_res]
plt.plot(lambdas_res, T, "-o")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmission")
plt.title("Stage-3 Ultra-High-Q Resonance")
plt.grid(True)
plt.show()
