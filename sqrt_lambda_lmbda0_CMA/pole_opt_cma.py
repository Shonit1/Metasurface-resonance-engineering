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

lambda0 = 1.6                      # target pole (µm)
lambdas = np.linspace(1.585, 1.615, 9)  # narrow local window

theta = 0
phi = 0

# --- OPTIMIZATION FIDELITY (FAST) ---
nG = 101
Nx = Ny = 300

# --- MATERIALS ---
eair = 1.0
esio2 = 1.44**2
hs = 0.1                           # Si thickness per DBR layer
num_pairs = 0                    # suppress FP background

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
    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    nV = obj.nG

    if np.abs(ai[k0]) > np.abs(ai[k0 + nV]):
        a00 = ai[k0]
        b00 = bi[k0]
    else:
        a00 = ai[k0 + nV]
        b00 = bi[k0 + nV]

    return b00 / a00

# ============================================================
# ---------------- PHASE SPECTRUM ----------------------------
# ============================================================

def compute_phase(r, h, hsi, a):
    ph = []
    for lam in lambdas:
        r00 = reflection_coeff(lam, r, h, hsi, a)
        ph.append(np.angle(r00))
    ph = np.unwrap(np.array(ph))
    ph -= ph[np.argmin(np.abs(lambdas - lambda0))]
    return ph

# ============================================================
# -------------- SIGNED SQRT TARGET --------------------------
# ============================================================

def signed_sqrt_dispersion(lam):
    out = np.zeros_like(lam)
    left = lam < lambda0
    right = lam > lambda0
    out[left] = -np.sqrt(lambda0 - lam[left])
    out[right] =  np.sqrt(lam[right] - lambda0)
    return out

# ============================================================
# ------------------- OBJECTIVE ------------------------------
# ============================================================

LOCAL_SLOPE_WEIGHT = 0.3   # increase to 0.4 later if stable

def objective(x):
    r, h, hsi = x
    a = a_fixed

    # ---------------- Hard constraints ----------------
    if not (0.05 < r < 0.45): return 1e4
    if not (0.05 < h < 1.2):  return 1e4
    if not (0.05 < hsi < 1.5): return 1e4

    try:
        ph = compute_phase(r, h, hsi, a)
    except:
        return 1e4

    # ---------------- Signed sqrt target ----------------
    target = signed_sqrt_dispersion(lambdas)
    A = np.dot(ph, target) / np.dot(target, target)
    fit_err = np.sqrt(np.mean((ph - A * target)**2))

    # ---------------- Derivatives ----------------
    dphi = np.gradient(ph, lambdas)
    ddphi = np.gradient(dphi, lambdas)

    idx0 = np.argmin(np.abs(lambdas - lambda0))

    # ---------------- Local pole enforcement ----------------
    local_slope = np.abs(dphi[idx0])

    # ---------------- Controlled re-coupling ----------------
    # reward strong slope at λ0
    slope_reward = -0.35 * local_slope

    # penalize shifted poles
    neighbor_slope = np.mean(np.abs(dphi[max(idx0-2,0):idx0+3]))
    shift_penalty = 0.08 * neighbor_slope

    # ---------------- Suppress multi-mode cheating ----------------
    curvature_penalty = 0.04 * np.mean(ddphi**2)

    # ---------------- Prevent weak cut-off ----------------
    # ensure phase actually accumulates (not just π jump)
    phase_span = np.max(ph) - np.min(ph)
    weak_coupling_penalty = 0.15 * np.exp(-phase_span)

    # ---------------- Final cost ----------------
    cost = (
        fit_err
        + slope_reward
        + shift_penalty
        + curvature_penalty
        + weak_coupling_penalty
    )

    return cost


# ============================================================
# -------- FIX LATTICE (POLE ALREADY PINNED) -----------------
# ============================================================

a_fixed = 0.9018    # <-- use your best a from previous run

# ============================================================
# ------------------- CMA-ES SETUP ---------------------------
# ============================================================

x0 = [0.4025, 0.7542, 0.2441]   # r, h, hSiO2
sigma0 = 0.08

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": [[0.05, 0.05, 0.05],
                   [0.45, 1.2, 1.5]],
        "popsize": 6,
        "maxiter": 12,
        "verb_disp": 0
    }
)

# ============================================================
# ------------------- OPTIMIZATION LOOP ----------------------
# ============================================================

print("\n--- LOCAL POLE REFINEMENT STARTED ---\n")

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
        f"mean={np.mean(F):.3e} | "
        f"time={time.time()-t0:.2f}s"
    )

r_opt, h_opt, hsi_opt = es.result.xbest

print("\n--- OPTIMIZED GEOMETRY ---")
print(f"r     = {r_opt:.4f} µm")
print(f"h     = {h_opt:.4f} µm")
print(f"hSiO2 = {hsi_opt:.4f} µm")
print(f"a     = {a_fixed:.4f} µm")

# ============================================================
# ------------------- FINAL DIAGNOSTICS ----------------------
# ============================================================

phis = compute_phase(r_opt, h_opt, hsi_opt, a_fixed)
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
plt.plot(lambdas, gd, 'o-', label="Group delay dφ/dλ")
plt.axvline(lambda0, color='r', linestyle='--')
plt.xlabel("Wavelength (µm)")
plt.ylabel("dφ/dλ (rad/µm)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
