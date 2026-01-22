import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import grcwa
import cma
import time

# ============================================================
# -------------------- GLOBAL PARAMETERS ---------------------
# ============================================================

lambda0 = 1.6
gamma = 0.01                 # complex regulator (NOT material loss)
lambdas = np.linspace(1.4, 1.8, 18)

theta = 0
phi = 0

# --- FAST OPTIMIZATION SETTINGS ---
nG = 101
Nx = Ny = 300

# --- MATERIALS ---
eair = 1.0
esio2 = 1.44**2
hs = 0.1
num_pairs = 1

# ============================================================
# ---------------- MATERIAL INTERPOLATION --------------------
# ============================================================

def epsilon_lambda(lam, _cache={}):
    if "interp" not in _cache:
        data = pd.read_csv(
            "C:\\Users\\ASUS\\Downloads\\Li-293K.csv"
        )
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values
        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["interp"](lam)**2

# ============================================================
# ----------------- GEOMETRY (CYLINDER) ----------------------
# ============================================================

def get_epgrid_cylinder(r, eps, a):
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xc = X - a/2
    Yc = Y - a/2
    mask = (Xc**2 + Yc**2) < r**2
    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps
    return ep.flatten()

# ============================================================
# --------------------- RCWA SOLVER --------------------------
# ============================================================

def reflection_coeff(lam, r, h, hsi, a):
    f = 1 / lam
    eps_si = epsilon_lambda(lam)

    L1 = [a, 0]
    L2 = [0, a]

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsi, esio2)

    for _ in range(num_pairs):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsi, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    epgrid = get_epgrid_cylinder(r, eps_si, a)
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(0, 0.0)
    nV = obj.nG
    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]

    # ---- CORRECT p-POLARIZED EXTRACTION ----
    a00 = ai[k0 + nV]
    b00 = bi[k0 + nV]

    return b00 / a00

# ============================================================
# ---------------- COMPLEX TARGET FUNCTION -------------------
# ============================================================

def complex_sqrt_target(lam):
    return np.sqrt((lam - lambda0) + 1j * gamma)

# ============================================================
# ------------------- OBJECTIVE FUNCTION ---------------------
# ============================================================

def objective(x):
    r, h, hsi, a = x

    # --- bounds ---
    if not (0.05 < r < 0.45): return 1e4
    if not (0.05 < h < 1.2): return 1e4
    if not (0.05 < hsi < 1.5): return 1e4
    if not (0.6 < a < 1.4): return 1e4

    try:
        rvals = np.array([
            reflection_coeff(lam, r, h, hsi, a)
            for lam in lambdas
        ])
    except:
        return 1e4

    target = complex_sqrt_target(lambdas)

    # Best complex scaling factor
    C = np.vdot(rvals, target) / np.vdot(target, target)

    err = np.mean(np.abs(rvals - C * target)**2)

    return err.real

# ============================================================
# ------------------- CMA-ES SETUP ---------------------------
# ============================================================

x0 = [0.25, 0.4, 0.3, 1.0]
sigma0 = 0.15

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": [[0.05, 0.05, 0.05, 0.6],
                   [0.45, 1.2, 1.5, 1.4]],
        "popsize": 8,
        "maxiter": 25,
        "verb_disp": 1
    }
)

# ============================================================
# ------------------- OPTIMIZATION LOOP ----------------------
# ============================================================

print("\n--- COMPLEX POLE SEARCH STARTED ---\n")

it = 0
while not es.stop():
    t0 = time.time()

    X = es.ask()
    F = [objective(x) for x in X]
    es.tell(X, F)

    it += 1
    print(
        f"Iter {it:02d} | "
        f"best={es.best.f:.3e} | "
        f"time={time.time()-t0:.2f}s"
    )

r_opt, h_opt, hsi_opt, a_opt = es.result.xbest

print("\n--- OPTIMIZED GEOMETRY ---")
print(f"r     = {r_opt:.4f} µm")
print(f"h     = {h_opt:.4f} µm")
print(f"hSiO2 = {hsi_opt:.4f} µm")
print(f"a     = {a_opt:.4f} µm")

# ============================================================
# ------------------- FINAL DIAGNOSTICS ----------------------
# ============================================================

rvals = np.array([
    reflection_coeff(lam, r_opt, h_opt, hsi_opt, a_opt)
    for lam in lambdas
])

phis = np.unwrap(np.angle(rvals))
gd = np.gradient(phis, lambdas)

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="Phase")
plt.axvline(lambda0, color='r', linestyle='--')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, gd, 'o-', label="Group delay")
plt.axvline(lambda0, color='r', linestyle='--')
plt.xlabel("Wavelength (µm)")
plt.ylabel("dφ/dλ (rad/µm)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
