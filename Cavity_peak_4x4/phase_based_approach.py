import numpy as np
import matplotlib.pyplot as plt
import cma
import grcwa
import pandas as pd
from scipy.interpolate import interp1d
from config import *

# =========================================================
# TARGET POLE LOCATION
# =========================================================
lambda0 = 1.5  # µm

# =========================================================
# WAVELENGTH GRID (DENSE NEAR POLE)
# =========================================================
lambdas = np.concatenate([
    np.linspace(1.49990, 1.49997, 7),
    np.linspace(1.49997, 1.50003, 14),
    np.linspace(1.50003, 1.50005, 7)
])

# =========================================================
# FIXED GEOMETRY
# =========================================================
P1 = np.array([
    1, 1, 1,
    0, 1, 0,
    0, 1, 1
]).reshape(3, 3)

P2 = np.array([
    0, 1, 1,
    0, 1, 1,
    1, 1, 0
]).reshape(3, 3)

A_FIXED = 1.2095

h1_init    = 0.2341
h2_init    = 0.1650
hs_init    = 0.2634
hsio2_init = 0.4000

# =========================================================
# TOP-K STORAGE
# =========================================================
TOP_K = 10
top_results = []

# =========================================================
# DISPERSION
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
# GRID BUILDER
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = (np.array(pattern).reshape(3,3) > 0.5).astype(int)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[(X >= i*dx) & (X < (i+1)*dx) &
                   (Y >= j*dy) & (Y < (j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_r00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)

        obj = grcwa.obj(
            nG,
            [A_FIXED, 0],
            [0, A_FIXED],
            1 / lam,
            theta,
            phi,
            verbose=0
        )

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1.flatten(), eps_si, A_FIXED).flatten()
        ep2 = get_epgrid_3x3(P2.flatten(), eps_si, A_FIXED).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1, 0, 0, 0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
        k0 = np.where((obj.G[:,0] == 0) & (obj.G[:,1] == 0))[0][0]
        nV = obj.nG

        r = bi[k0] / ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) else bi[k0+nV] / ai[k0+nV]

        if not np.isfinite(r):
            return None
        return r

    except Exception:
        return None
    




def solver_t00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)

        obj = grcwa.obj(
            nG, [A_FIXED,0], [0,A_FIXED],
            1/lam, theta, phi, verbose=0
        )

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1.flatten(), eps_si, A_FIXED).flatten()
        ep2 = get_epgrid_3x3(P2.flatten(), eps_si, A_FIXED).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)
        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        return Ti[k0]   

    except Exception:
        return None


# =========================================================
# PHASE
# =========================================================
def compute_phase(params):
    if params is None or len(params) != 4:
        return None

    phis = []
    for lam in lambdas:
        r00 = solver_r00(lam, *params)
        if r00 is None:
            return None
        phis.append(np.angle(r00))

    phis = np.unwrap(np.array(phis))
    if not np.all(np.isfinite(phis)):
        return None

    return phis

# =========================================================
# SQRT TARGET
# =========================================================
def sqrt_target(lams):
    return np.sqrt(np.maximum(lams - lambda0, 1e-6))

# =========================================================
# LOSS FUNCTION (CMA-SAFE + TOP-K)
# =========================================================
eval_counter = 0

def loss_phase_pole(x, w_fit=1.0, w_pole=4.0, w_amp=0.1):
    global eval_counter, top_results
    eval_counter += 1

    phis = compute_phase(x)
    if phis is None:
        return 1e6

    idx0 = np.argmin(np.abs(lambdas - lambda0))
    phis -= phis[idx0]

    mask = np.abs(lambdas - lambda0) > 2e-5
    if np.sum(mask) < 3:
        return 1e6

    tgt = sqrt_target(lambdas[mask])
    A = np.dot(phis[mask], tgt) / np.dot(tgt, tgt)
    fit_error = np.mean((phis[mask] - A * tgt)**2)

    if idx0 == 0 or idx0 == len(lambdas) - 1:
        return 1e6

    pole_strength = -abs(
        (phis[idx0+1] - phis[idx0-1]) /
        (lambdas[idx0+1] - lambdas[idx0-1])
    )

    amp_penalty = 0.0
    for lam in lambdas:
        r00 = solver_r00(lam, *x)
        if r00 is None:
            return 1e6
        amp_penalty += max(0, abs(r00) - 1.2)**2

    loss = w_fit * fit_error + w_pole * pole_strength + w_amp * amp_penalty
    if not np.isfinite(loss):
        return 1e6

    entry = {"loss": loss, "params": x.copy(), "phis": phis.copy()}
    top_results.append(entry)
    top_results = sorted(top_results, key=lambda e: e["loss"])[:TOP_K]

    print(
        f"[{eval_counter}] "
        f"h1={x[0]:.4f}, h2={x[1]:.4f}, hs={x[2]:.4f}, hsio2={x[3]:.4f} | "
        f"fit={fit_error:.2e}, |dφ/dλ|={abs(pole_strength):.2e}, LOSS={loss:.2e}"
    )

    return loss

# =========================================================
# CMA SETUP
# =========================================================
x0 = np.array([h1_init, h2_init, hs_init, hsio2_init])

bounds = [
    [h1_init * 0.8, h1_init * 1.2],
    [h2_init * 0.8, h2_init * 1.2],
    [hs_init * 0.9, hs_init * 1.1],
    [hsio2_init * 0.8, hsio2_init * 1.2],
]

es = cma.CMAEvolutionStrategy(
    x0,
    0.03,
    {
        "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
        "popsize": 6,
        "maxiter": 50,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_phase_pole(x) for x in xs])

# =========================================================
# FINAL TOP-10 REPORT
# =========================================================
print("\n===== TOP 10 PHASE-POLE DESIGNS =====")
for i, e in enumerate(top_results):
    h1, h2, hs, hsio2 = e["params"]
    print(
        f"[{i+1:02d}] LOSS={e['loss']:.3e} | "
        f"h1={h1:.6f}, h2={h2:.6f}, "
        f"hs={hs:.6f}, hsio2={hsio2:.6f}"
    )

# =========================================================
# PLOTS
# =========================================================
# =========================================================
# PER-GEOMETRY PHASE + TRANSMISSION PLOTS
# =========================================================
for i, e in enumerate(top_results):

    params = e["params"]
    phis = e["phis"]

    # -----------------------------
    # PHASE PLOT
    # -----------------------------
    plt.figure(figsize=(7,5))
    plt.plot(lambdas, phis, marker='o')
    plt.axvline(lambda0, color='k', linestyle=':')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Phase (rad)")
    plt.title(
        f"Geometry #{i+1} — Phase\n"
        f"h1={params[0]:.4f}, h2={params[1]:.4f}, "
        f"hs={params[2]:.4f}, hsio2={params[3]:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # TRANSMISSION PLOT
    # -----------------------------
    Ts = []
    for lam in lambdas:
        t = solver_t00(lam, *params)
        Ts.append(t if t is not None else 0.0)

    plt.figure(figsize=(7,5))
    plt.plot(lambdas, Ts, marker='o')
    plt.axvline(lambda0, color='k', linestyle=':')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transmission |t₀₀|²")
    plt.title(f"Geometry #{i+1} — Transmission")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # OPTIONAL: PHASE SLOPE PLOT
    # -----------------------------
    dphi = np.gradient(phis, lambdas)

    plt.figure(figsize=(7,5))
    plt.plot(lambdas, dphi, marker='o')
    plt.axvline(lambda0, color='k', linestyle=':')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("dφ/dλ (rad/µm)")
    plt.title(f"Geometry #{i+1} — Phase Slope")
    plt.grid(True)
    plt.tight_layout()
    plt.show()