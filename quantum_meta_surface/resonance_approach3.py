import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# STORE TOP CANDIDATES
# =========================================================
TOP_K = 10
top_candidates = []   # list of (loss, params)

def update_top_candidates(loss, params):
    global top_candidates
    top_candidates.append((loss, params.copy()))
    top_candidates.sort(key=lambda x: x[0])
    if len(top_candidates) > TOP_K:
        top_candidates = top_candidates[:TOP_K]

# =========================================================
# TARGET POLE LOCATIONS (µm)
# =========================================================
lambda0_A1 = 1.5
lambda0_A2 = 0.75

# =========================================================
# WAVELENGTH GRIDS
# =========================================================
lambdas_A1 = np.concatenate([
    np.linspace(1.485, 1.495, 6),
    np.linspace(1.495, 1.510, 16),
    np.linspace(1.510, 1.520, 6)
])

lambdas_A2 = np.concatenate([
    np.linspace(0.730, 0.740, 6),
    np.linspace(0.740, 0.755, 16),
    np.linspace(0.755, 0.765, 6)
])



# =========================================================
# MATERIAL MODEL ε(λ) — Aspnes (visible) + Li (IR)
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    """
    wavelength in microns
    returns complex permittivity ε = n^2
    """

    if "interp_li" not in _cache:
        # ---- Li 293K (IR) ----
        data_li = pd.read_csv(
            "C:\\Users\\ASUS\\Downloads\\Li-293K.csv"
        )
        wl_li = data_li.iloc[:, 0].values
        n_li  = data_li.iloc[:, 1].values

        _cache["interp_li"] = interp1d(
            wl_li, n_li,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    if "interp_aspnes" not in _cache:
        # ---- Aspnes (Visible–NIR) ----
        data_as = pd.read_csv(
            "C:\\Users\\ASUS\\Downloads\\Aspnes.csv"
        )
        wl_as = data_as.iloc[:, 0].values
        n_as  = data_as.iloc[:, 1].values

        _cache["interp_aspnes"] = interp1d(
            wl_as, n_as,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    # -----------------------------------------------------
    # Dataset switching logic
    # -----------------------------------------------------
    wavelength = np.asarray(wavelength)

    n_val = np.zeros_like(wavelength, dtype=float)

    # Visible / 0.7 µm band → Aspnes
    mask_vis = wavelength < 1.0
    n_val[mask_vis] = _cache["interp_aspnes"](wavelength[mask_vis])

    # IR / 1.5 µm band → Li
    mask_ir = ~mask_vis
    n_val[mask_ir] = _cache["interp_li"](wavelength[mask_ir])

    return n_val**2


# =========================================================
# 3×3 ε-GRID
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    pattern = np.clip(pattern.reshape(3,3), 0.05, 0.95)
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair

    dx = a / 3
    dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            mask = (
                (X >= i*dx) & (X < (i+1)*dx) &
                (Y >= j*dy) & (Y < (j+1)*dy)
            )
            epgrid[mask] = eair + pattern[i,j] * (et - eair)

    return epgrid

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, pattern, h, hsio2, a):
    try:
        L1 = [a, 0]
        L2 = [0, a]
        eps_mat = epsilon_lambda(1/f)

        obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_mat)
            obj.Add_LayerUniform(hsio2, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        epgrid = get_epgrid_3x3(pattern, eps_mat, a).flatten()
        obj.GridLayer_geteps(epgrid)

        obj.MakeExcitationPlanewave(
            p_amp=1, p_phase=0,
            s_amp=0, s_phase=0,
            order=0
        )

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        if abs(ai[k0]) > abs(ai[k0+nV]):
            return bi[k0] / ai[k0]
        else:
            return bi[k0+nV] / ai[k0+nV]

    except Exception:
        return None

# =========================================================
# PHASE
# =========================================================
def compute_phase(params, lambdas):
    pattern = params[:9]
    h, hsio2, a = params[9:]

    phis = []
    for lam in lambdas:
        r = solver_system(1/lam, pattern, h, hsio2, a)
        if r is None or not np.isfinite(r):
            return None
        phis.append(np.angle(r))

    return np.unwrap(np.array(phis))

# =========================================================
# R / T
# =========================================================
def compute_RT(params, lambdas):
    R, T = [], []
    for lam in lambdas:
        r = solver_system(1/lam, params[:9], params[9], params[10], params[11])
        if r is None:
            R.append(np.nan)
            T.append(np.nan)
        else:
            R.append(abs(r)**2)
            T.append(1 - abs(r)**2)
    return np.array(R), np.array(T)

# =========================================================
# BOUNDS
# =========================================================
bounds = np.array(
    [[0,1]]*9 +
    [[0.05,0.40],
     [0.05,0.60],
     [0.2,1.5]]
)

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# SQRT TARGET
# =========================================================
def sqrt_target(lams, lam0):
    eps = 1e-4
    return np.sqrt(np.maximum(lams - lam0, eps))

# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_double_sqrt_pole(x):

    params = decode(x)

    phi1 = compute_phase(params, lambdas_A1)
    phi2 = compute_phase(params, lambdas_A2)

    if phi1 is None or phi2 is None:
        return 1e6

    phi1 -= phi1[np.argmin(abs(lambdas_A1 - lambda0_A1))]
    phi2 -= phi2[np.argmin(abs(lambdas_A2 - lambda0_A2))]

    mask1 = lambdas_A1 > lambda0_A1 + 0.003
    mask2 = lambdas_A2 > lambda0_A2 + 0.003

    tgt1 = sqrt_target(lambdas_A1[mask1], lambda0_A1)
    tgt2 = sqrt_target(lambdas_A2[mask2], lambda0_A2)

    A1 = np.dot(phi1[mask1], tgt1) / np.dot(tgt1, tgt1)
    A2 = np.dot(phi2[mask2], tgt2) / np.dot(tgt2, tgt2)

    fit1 = np.mean((phi1[mask1] - A1*tgt1)**2)
    fit2 = np.mean((phi2[mask2] - A2*tgt2)**2)

    dphi1 = np.gradient(phi1, lambdas_A1)
    dphi2 = np.gradient(phi2, lambdas_A2)

    p1 = -abs(dphi1[np.argmin(abs(lambdas_A1 - lambda0_A1))])
    p2 = -abs(dphi2[np.argmin(abs(lambdas_A2 - lambda0_A2))])

    loss = fit1 + fit2 + 3*(p1+p2)

    update_top_candidates(loss, params)

    print(
    f"LOSS={loss:.3e} | "
    f"p1={p1:.3e}, p2={p2:.3e} | "
    f"A1={A1:.3e}, A2={A2:.3e}"
)

    return loss

# =========================================================
# CMA SETUP
# =========================================================
x0_phys = np.array([0.5]*9 + [0.15, 0.35, 1.2])
x0 = (x0_phys - bounds[:,0])/(bounds[:,1]-bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize":10, "maxiter":60, "verb_disp":1}
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_double_sqrt_pole(x) for x in xs])

# =========================================================
# FINAL PLOTS
# =========================================================
print("\n================ TOP 10 GEOMETRIES ================\n")

for i, (loss, params) in enumerate(top_candidates):

    print(f"\n--- Candidate {i+1} | LOSS = {loss:.3e} ---")
    print(params[:9].reshape(3,3))
    print(f"h={params[9]:.3f}, hsio2={params[10]:.3f}, a={params[11]:.3f}")

    phi1 = compute_phase(params, lambdas_A1)
    phi2 = compute_phase(params, lambdas_A2)

    lambdas_full = np.linspace(1.45, 1.70, 120)
    R, T = compute_RT(params, lambdas_full)

    fig, axs = plt.subplots(1, 2, figsize=(12,4))

    axs[0].plot(lambdas_A1, phi1, 'o-', label="A1")
    axs[0].plot(lambdas_A2, phi2, 's-', label="A2")
    axs[0].axvline(lambda0_A1, ls=":", c="r")
    axs[0].axvline(lambda0_A2, ls=":", c="r")
    axs[0].set_title("Phase vs Wavelength")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(lambdas_full, R, label="R")
    axs[1].plot(lambdas_full, T, label="T")
    axs[1].set_title("Reflection / Transmission")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()
