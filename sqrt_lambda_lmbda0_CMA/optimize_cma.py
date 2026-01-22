import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import grcwa
import cma
import time


# ============================================================
# ------------------- GLOBAL PARAMETERS ----------------------
# ============================================================

lambda0 = 1.6          # pole location (µm)
lmin, lmax = 1.4, 1.8  # fitting window
lambdas = lambdas = np.linspace(1.45, 1.85, 35)

theta = 0
phi = 0
nG = 101
Nx, Ny = 300, 300

eair = 1.0
esio2 = 1.44**2
hs = 0.1

# ============================================================
# ------------------- MATERIAL MODEL -------------------------
# ============================================================

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

# ============================================================
# ------------------- GEOMETRY GRID --------------------------
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
# ------------------- RCWA SOLVER ----------------------------
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

    for _ in range(4):
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


def signed_sqrt_dispersion(lam, lambda0):
    out = np.zeros_like(lam)
    left = lam < lambda0
    right = lam > lambda0
    out[left] = -np.sqrt(lambda0 - lam[left])
    out[right] =  np.sqrt(lam[right] - lambda0)
    return out

    

# ============================================================
# ------------------- PHASE SPECTRUM -------------------------
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
# ------------------- OBJECTIVE FUNCTION ---------------------
# ============================================================

def objective(x):
    r, h, hsi, a = x

    # ---- hard constraints ----
    if not (0.05 < r < 0.45): return 1e4
    if not (0.05 < h < 1.2): return 1e4
    if not (0.05 < hsi < 1.5): return 1e4
    if not (0.6 < a < 1.4): return 1e4

    try:
        ph = compute_phase(r, h, hsi, a)
    except:
        return 1e4

    # ---- focus tightly around lambda0 ----
    mask = (lambdas > lambda0 - 0.08) & (lambdas < lambda0 + 0.08)
    lam = lambdas[mask]
    phi = ph[mask]

    # ---- signed sqrt target ----
    target = signed_sqrt_dispersion(lam, lambda0)

    # ---- best amplitude fit ----
    A = np.dot(phi, target) / np.dot(target, target)
    fit_err = np.sqrt(np.mean((phi - A * target)**2))

    # ---- enforce divergence AT lambda0 ----
    slope = np.gradient(ph, lambdas)
    idx0 = np.argmin(np.abs(lambdas - lambda0))

    local_slope = np.abs(slope[idx0])
    neighbor_slope = np.mean(np.abs(slope[idx0-2:idx0+3]))

    # ---- final cost ----
    cost = (
        fit_err
        - 0.1 * local_slope          # reward divergence at lambda0
        + 0.02 * neighbor_slope      # penalize shifted poles
    )

    return cost



    return cost

# ============================================================
# ------------------- CMA-ES OPTIMIZATION --------------------
# ============================================================

x0 = [0.25, 0.5, 0.3, 1.0]  # initial guess
sigma0 = 0.15

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": [[0.05, 0.005, 0.05, 0.6],
                   [0.45, 1.2, 1.5, 1.4]],
        "popsize": 6,
        "verb_disp": 1,
        "maxiter": 15
    }
)



iter_count = 0

while not es.stop():
    t0 = time.time()

    # Ask CMA for candidate solutions
    solutions = es.ask()

    # Evaluate objective for each solution
    costs = [objective(x) for x in solutions]

    # Update CMA-ES internal state
    es.tell(solutions, costs)
    es.logger.add()

    iter_count += 1
    dt = time.time() - t0

    # PRINT STATUS
    print(
        f"[CMA] iter={iter_count:02d} | "
        f"best={es.best.f:.4e} | "
        f"mean={np.mean(costs):.4e} | "
        f"sigma={es.sigma:.3e} | "
        f"time={dt:.2f}s"
    )

res = es.result.xbest
r_opt, h_opt, hsi_opt, a_opt = res

print("\n================= OPTIMAL DESIGN =================")
print(f"r      = {r_opt:.4f} µm")
print(f"h      = {h_opt:.4f} µm")
print(f"hSiO2  = {hsi_opt:.4f} µm")
print(f"a      = {a_opt:.4f} µm")

# ============================================================
# ------------------- FINAL PLOT -----------------------------
# ============================================================

phis = compute_phase(r_opt, h_opt, hsi_opt, a_opt)

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")
plt.axvline(lambda0, color='r', linestyle='--', label="λ₀ (pole)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
