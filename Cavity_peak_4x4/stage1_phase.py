import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

'''Resonance tuning with 4×4 patterned metasurface'''

# =========================================================
# WAVELENGTH WINDOWS
# =========================================================
lambdas_L = np.linspace(1.49990, 1.49995, 7)
lambdas_C = np.linspace(1.49995, 1.50000, 7)
lambdas_R = np.linspace(1.50000, 1.50005, 7)
lambdas_all = np.concatenate([lambdas_L, lambdas_C, lambdas_R])

lam0 = 1.5
dlam = 1e-4
lam_norm = (lambdas_all - lam0) / dlam

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
    n = 3.7 if wavelength < 1.0 else _cache["li"](wavelength)
    return n**2

# =========================================================
# 4×4 GRID
# =========================================================
def get_epgrid_4x4(pattern, eps, a):
    pattern = np.array(pattern).reshape(4,4)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/4
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(4):
        for j in range(4):
            fill = np.clip(pattern[i,j], 0.0, 1.0)
            mask = (
                (X >= i*dx) & (X < (i+1)*dx) &
                (Y >= j*dy) & (Y < (j+1)*dy)
            )
            ep[mask] = fill*eps + (1-fill)*eair

    return ep

# =========================================================
# RCWA SOLVER (R00 PHASE)
# =========================================================
def solver_r00(lam, pattern, h1, hsio2, hs, a):
    try:
        eps_si = epsilon_lambda(lam)

        obj = grcwa.obj(
            nG,
            [a,0],
            [0,a],
            1/lam,
            theta,
            phi,
            verbose=0
        )

        # Air
        obj.Add_LayerUniform(0.1, eair)

        # Patterned metasurface
        obj.Add_LayerGrid(h1, Nx, Ny)

        # SiO2 spacer
        obj.Add_LayerUniform(hsio2, esio2)

        # DBR
        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        # glass
        obj.Add_LayerUniform(0.1, esio2)

        obj.Init_Setup()

        ep = get_epgrid_4x4(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1, 0, 0, 0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

        k0 = np.where(
            (obj.G[:,0] == 0) & (obj.G[:,1] == 0)
        )[0][0]

        nV = obj.nG
        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        return r if np.isfinite(r) else None

    except Exception:
        return None

# =========================================================
# PHASE COMPUTATION
# =========================================================
def compute_phase(params):
    pattern = params[:16]
    h1, hsio2, hs, a = params[16:]

    phis = []
    for lam in lambdas_all:
        r = solver_r00(lam, pattern, h1, hsio2, hs, a)
        if r is None:
            return None
        phis.append(np.angle(r))

    return np.unwrap(np.array(phis))

# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0,1]]*16 +      # 4×4 pattern
    [
        [0.08,0.35],  # h1 (pattern height)
        [0.05,0.40],  # hsio2 (spacer)
        [0.20,0.28],  # hs (DBR Si)
        [0.50,1.5]   # a (period)
    ]
)

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# TOP-K STORAGE
# =========================================================
TOP_K = 10
top = []

def update_top(loss, params):
    if not np.isfinite(loss):
        return
    top.append((loss, params.copy()))
    top.sort(key=lambda x: x[0])
    del top[TOP_K:]

# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_function(x, alpha=0):
    params = decode(x)
    phis = compute_phase(params)

    if phis is None:
        return 1e6

    A, B = np.polyfit(lam_norm, phis, 1)
    rms = np.sqrt(np.mean((phis - (A*lam_norm + B))**2))

    loss = -abs(A) 

    print(
        f"A(norm)={A:6.2f}, RMS={rms:.2e}, a={params[-1]:.3f}"
        f"loss={loss:.3e}"
    )

    update_top(loss, params)
    return loss

# =========================================================
# CMA-ES OPTIMIZATION
# =========================================================
x0 = np.full(len(bounds), 0.5)

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize": 10, "maxiter": 40}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])

# =========================================================
# FINAL REPLAY
# =========================================================
for i, (loss, params) in enumerate(top):
    pattern = params[:16].reshape(4,4)
    h1, hsio2, hs, a = params[16:]

    print("\n" + "="*60)
    print(f"GEOMETRY #{i}")
    print(f"LOSS = {loss:.3e}")
    print("4×4 Pattern:\n", (pattern > 0.5).astype(int))
    print(f"h1={h1:.4f}, hsio2={hsio2:.4f}, hs={hs:.4f}, a={a:.4f}")

    phis = compute_phase(params)

    plt.plot(lambdas_all, phis, 'o-')
    plt.axvline(1.5, color='r', linestyle='--')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title(f"Stage-1 Phase Slope #{i}")
    plt.grid(True)
    plt.show()
