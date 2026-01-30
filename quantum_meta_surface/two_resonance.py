import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# USER TARGETS
# =========================================================
PHASE_SWING_TARGET = np.pi
SWING_TOL = 0.10          # acceptable error (rad)
LOCKED = False
BEST_PARAMS = None

# =========================================================
# Wavelength bands (µm)
# =========================================================
lambdas_A1 = np.linspace(1.52, 1.25, 8)
lambdas_A2 = np.linspace(1.62, 1.65, 8)

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
# RCWA SOLVER (FAIL-SAFE)
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
# PHASE + AMPLITUDE
# =========================================================
def compute_phase_and_amp(params, lambdas):
    pattern = params[:9]
    h, hsio2, a = params[9:]

    phis, amps = [], []
    for lam in lambdas:
        r = solver_system(1/lam, pattern, h, hsio2, a)
        if r is None or not np.isfinite(r):
            return None, None
        phis.append(np.angle(r))
        amps.append(abs(r))

    return np.unwrap(np.array(phis)), np.array(amps)

# =========================================================
# PARAMETER BOUNDS + DECODE
# =========================================================
bounds = np.array(
    [[0,1]]*9 + [[0.05,0.40],[0.05,0.60],[0.45,0.80]]
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# LOSS FUNCTION: π-SWING TARGETING
# =========================================================
def loss_function(x):
    global LOCKED, BEST_PARAMS

    params = decode(x)

    phi1,_ = compute_phase_and_amp(params, lambdas_A1)
    phi2,_ = compute_phase_and_amp(params, lambdas_A2)

    if phi1 is None or phi2 is None:
        return 1e6

    swing1 = np.max(phi1) - np.min(phi1)
    swing2 = np.max(phi2) - np.min(phi2)

    err1 = swing1 - PHASE_SWING_TARGET
    err2 = swing2 - PHASE_SWING_TARGET

    print(f"swing1={swing1:.2f}, swing2={swing2:.2f}", end=" ")

    # === LOCK WHEN BOTH ≈ π ===
    if abs(err1) < SWING_TOL and abs(err2) < SWING_TOL:
        if not LOCKED:
            print(">>> π-SWING ACHIEVED IN BOTH BANDS — LOCKING <<<")
            BEST_PARAMS = params.copy()
            LOCKED = True
        else:
            print(">>> LOCKED <<<")
        return -1e6

    if LOCKED:
        return -1e6

    # Quadratic penalty to π target
    loss = err1**2 + err2**2 + 0.5*(swing1-swing2)**2
    print(f"LOSS={loss:.3f}")
    return loss

# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array([0.5]*9 + [0.12,0.30,0.60])
x0 = (x0_phys - bounds[:,0])/(bounds[:,1]-bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize":12, "maxiter":60, "verb_disp":1}
)

# =========================================================
# CMA LOOP WITH HARD STOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_function(x) for x in xs]
    es.tell(xs, losses)
    if LOCKED:
        print("\n>>> CMA STOPPED: TARGET π-SWING GEOMETRY FOUND <<<")
        break

# =========================================================
# FINAL GEOMETRY
# =========================================================
best_params = BEST_PARAMS if BEST_PARAMS is not None else decode(es.result.xbest)

print("\n===== FINAL GEOMETRY =====")
print("Pixel pattern:")
print((best_params[:9]>0.5).reshape(3,3).astype(int))
print("h     =", best_params[9])
print("hsio2 =", best_params[10])
print("a     =", best_params[11])

# =========================================================
# FINAL PLOTS
# =========================================================
phi1,_ = compute_phase_and_amp(best_params, lambdas_A1)
phi2,_ = compute_phase_and_amp(best_params, lambdas_A2)

plt.figure(figsize=(6,4))
plt.plot(lambdas_A1, phi1, 'o-', label="Band A1")
plt.plot(lambdas_A2, phi2, 's-', label="Band A2")
plt.axhline(np.pi, color='k', ls='--', lw=0.8)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("Dual π-phase-swing resonant geometry")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
