import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import time

# =========================================================
# GLOBAL SETTINGS
# =========================================================
N_CAV = 7
N_PEAKS = 7
DBR_PAIRS = 5

PATTERN_N = 4
N_PATTERN = PATTERN_N * PATTERN_N

hsio2_dbr = 0.2586
dlam = 1e-4

lambda_targets = np.linspace(1.40, 1.50, N_PEAKS)

# =========================================================
# LOAD STAGE-1 SUCCESS FILE (4×4)
# =========================================================
STAGE1_FILE = "stage1_SUCCESS_min0p5_20260203_190218.npz"  # ← update

def load_stage1_npz(fname):
    data = np.load(fname)
    p = data["params"]

    pattern  = p[:N_PATTERN].copy()
    hs_list  = p[N_PATTERN : N_PATTERN + N_CAV].copy()
    hsp_list = p[N_PATTERN + N_CAV : N_PATTERN + 2*N_CAV].copy()
    hs_dbr   = p[N_PATTERN + 2*N_CAV]
    a        = p[N_PATTERN + 2*N_CAV + 1]

    print("\nLoaded Stage-1 geometry (4×4):")
    print(f"  a        = {a:.4f}")
    print(f"  hs_dbr   = {hs_dbr:.4f}")
    print(f"  mean hs  = {hs_list.mean():.4f}")
    print(f"  mean hsp = {hsp_list.mean():.4f}")

    return pattern, hs_list, hsp_list, hs_dbr, a

FIXED_PATTERN, hs_init, hsp_init, hs_dbr_init, a_init = load_stage1_npz(
    STAGE1_FILE
)

# =========================================================
# PARAMETER BOUNDS (STAGE-2)
# =========================================================
bounds = np.array(
    [[0.20, 0.45]] +           # h_pattern
    [[0.22, 0.28]] * N_CAV +   # hs_i
    [[0.15, 0.35]] * N_CAV +   # hsp_i
    [[0.22, 0.28]] +           # hs_dbr
    [[1.32, 1.36]]             # a
)

N_PARAMS = bounds.shape[0]

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:, 0], data.iloc[:, 1],
            kind="cubic", bounds_error=False,
            fill_value="extrapolate"
        )
    n = 3.7 if wavelength < 1.0 else _cache["li"](wavelength)
    return n**2

# =========================================================
# GENERIC N×N METASURFACE GRID
# =========================================================
def get_epgrid(pattern, eps, a, N):
    pattern = np.array(pattern).reshape(N, N)
    ep = np.ones((Nx, Ny), dtype=complex) * eair

    dx = dy = a / N
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(N):
        for j in range(N):
            f = np.clip(pattern[i, j], 0, 1)
            ep[
                (X >= i*dx) & (X < (i+1)*dx) &
                (Y >= j*dy) & (Y < (j+1)*dy)
            ] = f*eps + (1-f)*eair

    return ep

# =========================================================
# RCWA SOLVER
# =========================================================
def solve_rt(lam, h_pattern, hs_list, hsp_list, hs_dbr, a):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG, [a,0], [0,a], 1/lam, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h_pattern, Nx, Ny)

        for hs, hsp in zip(hs_list, hsp_list):
            obj.Add_LayerUniform(hsp, esio2)
            obj.Add_LayerUniform(hs, eps_si)

        for _ in range(DBR_PAIRS):
            obj.Add_LayerUniform(hs_dbr, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep = get_epgrid(FIXED_PATTERN, eps_si, a, PATTERN_N).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1, 0, 0, 0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        r = complex(r)

        if not np.isfinite(r.real) or not np.isfinite(r.imag):
            return None, None

        _, T = obj.RT_Solve(normalize=1, byorder=1)
        return r, T[k0]

    except Exception:
        return None, None

# =========================================================
# STAGE-2 LOSS (UNCHANGED PHYSICS)
# =========================================================
def stage2b_loss(x):
    p = decode(x)

    h_pattern = p[0]
    hs_list   = p[1:1+N_CAV]
    hsp_list  = p[1+N_CAV:1+2*N_CAV]
    hs_dbr    = p[-2]
    a         = p[-1]

    peaks = []
    curvatures = []

    for lam0 in lambda_targets:
        phis = []
        Ts   = []

        for lam in (lam0-dlam, lam0, lam0+dlam):
            r, T = solve_rt(lam, h_pattern, hs_list, hsp_list, hs_dbr, a)
            if r is None:
                return 1e6
            phis.append(np.angle(r))
            Ts.append(T)

        phis = np.unwrap(phis)
        d2phi = (phis[2] - 2*phis[1] + phis[0]) / dlam**2

        curvatures.append(d2phi)
        peaks.append(Ts[1])

    peaks = np.array(peaks)
    curvatures = np.array(curvatures)

    peaks_sorted = np.sort(peaks)
    k = max(1, int(0.3 * len(peaks)))
    T_tail = peaks_sorted[:k].mean()

    T_TARGET = 0.8
    alpha = 20.0

    tail_pull = np.log1p(np.exp(alpha * (T_TARGET - T_tail)))
    mean_pull = 0.1 * np.log1p(np.exp(10 * (T_TARGET - peaks.mean())))

    curv_norm = np.abs(curvatures)
    curv_balance = np.std(curv_norm) / (np.mean(curv_norm) + 1e-9)

    loss = (
        1.0 * tail_pull +
        0.2 * mean_pull +
        0.2 * curv_balance
    )

    print(
        f"Tmin={peaks.min():.3f}, "
        f"Tμ={peaks.mean():.3f}, "
        f"loss={loss:.3e}, "
        f"a={a:.3f}"
    )

    return loss

# =========================================================
# CMA-ES
# =========================================================
x0 = np.full(N_PARAMS, 0.5)

es = cma.CMAEvolutionStrategy(
    x0, 0.15, {"popsize":10, "maxiter":40}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [stage2b_loss(x) for x in xs])




# =========================================================
# EXTRACT BEST SOLUTION
# =========================================================
best_x = es.result.xbest
best_p = decode(best_x)

h_pattern = best_p[0]
hs_list   = best_p[1:1+N_CAV]
hsp_list  = best_p[1+N_CAV:1+2*N_CAV]
hs_dbr    = best_p[-2]
a         = best_p[-1]

print("\n" + "="*60)
print("STAGE-2 OPTIMIZATION COMPLETE (4×4)")
print("="*60)

print(f"h_pattern = {h_pattern:.4f}")
print(f"a         = {a:.4f}")
print(f"hs_dbr    = {hs_dbr:.4f}")

print("\nCAVITY THICKNESSES (Si):")
for i, h in enumerate(hs_list):
    print(f"  hs[{i}] = {h:.4f}")

print("\nSPACER THICKNESSES (SiO2):")
for i, h in enumerate(hsp_list):
    print(f"  hsp[{i}] = {h:.4f}")

print("\nFIXED PATTERN (4×4):")
print(FIXED_PATTERN.reshape(PATTERN_N, PATTERN_N))


# =========================================================
# FINAL DIAGNOSTIC SWEEP
# =========================================================
lams = np.linspace(1.38, 1.52, 600)

phis = np.zeros_like(lams)
T    = np.zeros_like(lams)
R    = np.zeros_like(lams)

for i, lam in enumerate(lams):
    r, t = solve_rt(lam, h_pattern, hs_list, hsp_list, hs_dbr, a)
    if r is None:
        phis[i] = np.nan
        T[i]    = np.nan
        R[i]    = np.nan
    else:
        phis[i] = np.angle(r)
        T[i]    = t
        R[i]    = abs(r)**2

phis = np.unwrap(phis)


# =========================================================
# CHECK PROFESSOR'S SUCCESS CONDITION
# =========================================================
T_peaks = []

for lam0 in lambda_targets:
    idx = np.argmin(np.abs(lams - lam0))
    T_peaks.append(T[idx])

T_peaks = np.array(T_peaks)

print("\nTRANSMISSION AT TARGET PEAKS:")
for i, (lam0, val) in enumerate(zip(lambda_targets, T_peaks)):
    print(f"  λ[{i}] = {lam0:.4f} µm → T = {val:.3f}")

print("\nMINIMUM PEAK TRANSMISSION:")
print(f"  Tmin = {T_peaks.min():.3f}")

if T_peaks.min() >= 0.8:
    print("✅ SUCCESS: All peaks satisfy T ≥ 0.8")
else:
    print("❌ NOT YET: Some peaks below 0.8")


# =========================================================
# PLOTS
# =========================================================
fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

ax[0].plot(lams, phis, lw=2)
ax[0].set_ylabel("Unwrapped phase (rad)")
ax[0].set_title("Final Stage-2 Response (4×4)")

ax[1].plot(lams, T, lw=2)
ax[1].set_ylabel("Transmission")

ax[2].plot(lams, R, lw=2)
ax[2].set_ylabel("Reflection")
ax[2].set_xlabel("Wavelength (µm)")

for lam in lambda_targets:
    for axt in ax:
        axt.axvline(lam, color="r", ls="--", alpha=0.35)

for axt in ax:
    axt.grid(True)

plt.tight_layout()
plt.show()


