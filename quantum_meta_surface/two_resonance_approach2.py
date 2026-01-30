import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# GLOBAL STORAGE FOR RESONANT GEOMETRIES
# =========================================================
RESONANT_GEOMS = []

# =========================================================
# Wavelength bands (µm)
# =========================================================
lambdas_A1 = np.linspace(1.52, 1.55, 10)
lambdas_A2 = np.linspace(1.62, 1.65, 10)

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
# 3×3 pixelated ε-grid (SMOOTH)
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    pattern = np.clip(np.array(pattern).reshape(3,3), 0.05, 0.95)

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
# RCWA SOLVER (FAIL SAFE)
# =========================================================
def solver_system(f, pattern, h, hsio2, a):
    try:
        L1 = [a, 0]
        L2 = [0, a]
        es = epsilon_lambda(1/f)

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
# PHASE COMPUTATION
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
# PARAMETER BOUNDS + DECODE
# =========================================================
bounds = np.array(
    [[0,1]]*9 + [[0.05,0.40],[0.05,0.60],[0.2,1.50]]
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# LOSS FUNCTION: MAXIMIZE A1 + A2
# =========================================================
def loss_function(x, balance_weight=0.2):
    params = decode(x)

    phi1 = compute_phase(params, lambdas_A1)
    phi2 = compute_phase(params, lambdas_A2)

    if phi1 is None or phi2 is None:
        return 1e6

    # Linear slopes
    A1, _ = np.polyfit(lambdas_A1, phi1, 1)
    A2, _ = np.polyfit(lambdas_A2, phi2, 1)

    # Phase swings (for resonance detection)
    swing1 = np.max(phi1) - np.min(phi1)
    swing2 = np.max(phi2) - np.min(phi2)

    # === SOFT RESONANCE RECORDING ===
    if swing1 > 1.5 and swing2 > 1.5:
        RESONANT_GEOMS.append({
            "params": params.copy(),
            "A1": A1,
            "A2": A2,
            "swing1": swing1,
            "swing2": swing2
        })
        print(">>> RESONANCE RECORDED <<<")

    # === LOSS ===
    balance_penalty = (A1 - A2)**2
    loss = -np.abs(A1) - np.abs(A2) + balance_weight * balance_penalty

    print(
        f"A1={A1:.2f}, A2={A2:.2f}, "
        f"|A1-A2|={abs(A1-A2):.2f}, LOSS={loss:.2f}"
    )

    return loss


# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array([0.5]*9 + [0.12,0.30,1.30])
x0 = (x0_phys - bounds[:,0])/(bounds[:,1]-bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize":12, "maxiter":60, "verb_disp":1}
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_function(x) for x in xs]
    es.tell(xs, losses)

# =========================================================
# POST-PROCESSING: FIND π–SWING GEOMETRY
# =========================================================
print("\n===== POST PROCESSING RESONANCES =====")
best = None
best_err = 1e9

for g in RESONANT_GEOMS:
    err = abs(g["swing1"]-np.pi) + abs(g["swing2"]-np.pi)
    if err < best_err:
        best_err = err
        best = g

if best is not None:
    best_params = best["params"]
    print("Best π-approx geometry found")
    print("swing1 =", best["swing1"])
    print("swing2 =", best["swing2"])
else:
    best_params = decode(es.result.xbest)
    print("No π geometry found, using CMA best")

# =========================================================
# FINAL OUTPUT
# =========================================================
print("\n===== FINAL GEOMETRY =====")
print((best_params[:9]>0.5).reshape(3,3).astype(int))
print("h =", best_params[9])
print("hsio2 =", best_params[10])
print("a =", best_params[11])

phi1 = compute_phase(best_params, lambdas_A1)
phi2 = compute_phase(best_params, lambdas_A2)

plt.figure(figsize=(6,4))
plt.plot(lambdas_A1, phi1, 'o-', label="Band A1")
plt.plot(lambdas_A2, phi2, 's-', label="Band A2")
plt.axhline(np.pi, ls="--", color="k")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
