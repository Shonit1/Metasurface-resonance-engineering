import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import time

# =========================================================
# GLOBAL SETTINGS (STAGE-2)
# =========================================================
N_CAV = 7
N_PEAKS = 7
DBR_PAIRS = 5

hsio2_dbr = 0.2586
dlam = 1e-4

lambda_targets = np.linspace(1.4, 1.5, N_PEAKS)

# =========================================================
# LOAD STAGE-1 SUCCESS FILE
# =========================================================
STAGE1_FILE = "stage1_SUCCESS_min0p5_20260202_152910.npz"  # ← update if needed

def load_stage1_npz(fname):
    data = np.load(fname)
    p = data["params"]

    pattern  = p[:9].copy()
    hs_list  = p[9 : 9 + N_CAV].copy()
    hsp_list = p[9 + N_CAV : 9 + 2*N_CAV].copy()
    hs_dbr   = p[9 + 2*N_CAV]
    a        = p[9 + 2*N_CAV + 1]

    print("\nLoaded Stage-1 geometry:")
    print(f"  a        = {a:.4f}")
    print(f"  hs_dbr   = {hs_dbr:.4f}")
    print(f"  mean hs  = {hs_list.mean():.4f}")
    print(f"  mean hsp = {hsp_list.mean():.4f}")
    print(f"  min/μ    = {data['ratio_min']:.3f}")

    return pattern, hs_list, hsp_list, hs_dbr, a

# =========================================================
# LOAD + FREEZE PATTERN
# =========================================================
FIXED_PATTERN, hs_init, hsp_init, hs_dbr_init, a_init = load_stage1_npz(
    STAGE1_FILE
)

# =========================================================
# STAGE-2 PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0.20,0.45]] +        # h_pattern
    [[0.22,0.28]]*N_CAV +  # hs_i
    [[0.15,0.35]]*N_CAV +  # hsp_i
    [[0.22,0.28]] +        # hs_dbr
    [[1.32,1.36]]          # a
)

N_PARAMS = bounds.shape[0]

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# BUILD WARM-START x0 FOR STAGE-2
# =========================================================
def build_stage2_x0():
    p_phys = np.concatenate([
        [0.30],        # initial h_pattern
        hs_init,
        hsp_init,
        [hs_dbr_init],
        [a_init]
    ])

    x0 = (p_phys - bounds[:,0]) / (bounds[:,1] - bounds[:,0])
    x0 = np.clip(x0, 0.0, 1.0)

    print("\nStage-2 warm start prepared:")
    print(f"  h_pattern_init = 0.30")
    print(f"  x0 dimension   = {len(x0)}")

    return x0

x0 = build_stage2_x0()



def save_stage2_success(params, peaks, loss):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"stage2_SUCCESS_T0p8_{ts}.npz"

    np.savez(
        fname,
        params=params,
        peaks=peaks,
        Tmin=peaks.min(),
        Tmean=peaks.mean(),
        loss=loss
    )

    print("\n" + "="*60)
    print("🎉 STAGE-2 SUCCESS: ALL PEAKS ≥ 0.8")
    print(f"Saved geometry → {fname}")
    print("="*60 + "\n")









# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0], data.iloc[:,1],
            kind="cubic", bounds_error=False,
            fill_value="extrapolate"
        )
    n = 3.7 if wavelength < 1.0 else _cache["li"](wavelength)
    return n**2

# =========================================================
# 3×3 METASURFACE
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = np.array(pattern).reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            f = np.clip(pattern[i,j],0,1)
            ep[(X>=i*dx)&(X<(i+1)*dx)&
               (Y>=j*dy)&(Y<(j+1)*dy)] = f*eps + (1-f)*eair
    return ep

# =========================================================
# RCWA SOLVER (r, T)
# =========================================================
def solve_rt(lam, h_pattern, hs_list, hsp_list, hs_dbr, a):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG,[a,0],[0,a],1/lam,theta,phi,verbose=0)

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

        ep = get_epgrid_3x3(FIXED_PATTERN, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        r = complex(r)

        if not (np.isfinite(r.real) and np.isfinite(r.imag)):
            return None, None

        _, T = obj.RT_Solve(normalize=1, byorder=1)
        return r, T[k0]

    except Exception:
        return None, None

# =========================================================
# PARAMETERS (STAGE-2)
# =========================================================
N_PARAMS = 1 + N_CAV + N_CAV + 1 + 1   # = 17

bounds = np.array(
    [[0.20,0.45]] +        # h_pattern
    [[0.22,0.28]]*N_CAV +  # hs_i
    [[0.15,0.35]]*N_CAV +  # hsp_i
    [[0.22,0.28]] +        # hs_dbr
    [[1.32,1.36]]          # a
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# STAGE-2 LOSS FUNCTION
# =========================================================
def stage2b_loss(x):

    # -----------------------------
    # Decode parameters
    # -----------------------------
    p = decode(x)

    h_pattern = p[0]
    hs_list   = p[1:1+N_CAV]
    hsp_list  = p[1+N_CAV:1+2*N_CAV]
    hs_dbr    = p[-2]
    a         = p[-1]

    peaks = []
    curvatures = []

    # -----------------------------
    # Evaluate resonances
    # -----------------------------
    for lam0 in lambda_targets:

        phis = []
        Ts   = []

        for lam in (lam0 - dlam, lam0, lam0 + dlam):
            r, T = solve_rt(
                lam,
                h_pattern,
                hs_list,
                hsp_list,
                hs_dbr,
                a
            )
            if r is None or T is None:
                return 1e6

            phis.append(np.angle(r))
            Ts.append(T)

        phis = np.unwrap(phis)
        d2phi = (phis[2] - 2*phis[1] + phis[0]) / dlam**2

        curvatures.append(d2phi)
        peaks.append(Ts[1])

    peaks = np.array(peaks)
    curvatures = np.array(curvatures)

    # =====================================================
    # (1) TAIL-FOCUSED TRANSMISSION PUSH  (MAIN DRIVER)
    # =====================================================

    # Sort peaks from worst to best
    peaks_sorted = np.sort(peaks)

    # Bottom 30% of modes
    k = max(1, int(0.3 * len(peaks)))
    T_tail = peaks_sorted[:k].mean()

    T_TARGET = 0.8
    alpha = 20.0

    tail_pull = np.log1p(np.exp(alpha * (T_TARGET - T_tail)))

    # =====================================================
    # (2) VERY WEAK MEAN REGULARIZATION
    # =====================================================
    mean_pull = 0.1 * np.log1p(np.exp(10 * (T_TARGET - peaks.mean())))

    # =====================================================
    # (3) GENTLE CURVATURE LEASH (DO NOT OVERCONSTRAIN)
    # =====================================================
    curv_norm = np.abs(curvatures)
    curv_balance = (
        np.std(curv_norm) /
        (np.mean(curv_norm) + 1e-9)
    )

    # =====================================================
    # TOTAL LOSS
    # =====================================================
    loss = (
        1.0 * tail_pull +
        0.2 * mean_pull +
        0.2 * curv_balance
    )

    # -----------------------------
    # Logging
    # -----------------------------
    print(
        f"Tmin={peaks.min():.3f}, "
        f"Ttail={T_tail:.3f}, "
        f"Tμ={peaks.mean():.3f}, "
        f"loss={loss:.3e}, "
        f"a={a:.3f}, hpat={h_pattern:.3f}"
    )

    # -----------------------------
    # Success save
    # -----------------------------
    if peaks.min() >= 0.8:
        save_stage2_success(p, peaks, loss)

    return loss



# =========================================================
# CMA-ES (STAGE-2)
# =========================================================
x0 = np.full(N_PARAMS,0.5)

es = cma.CMAEvolutionStrategy(
    x0,0.15,{"popsize":10,"maxiter":40}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs,[stage2b_loss(x) for x in xs])


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
print("STAGE-2 OPTIMIZATION COMPLETE")
print("="*60)

print(f"h_pattern = {h_pattern:.4f}")
print(f"a         = {a:.4f}")
print(f"hs_dbr    = {hs_dbr:.4f}")

print("\nCAVITY THICKNESSES (Si):")
for i,h in enumerate(hs_list):
    print(f"  hs[{i}] = {h:.4f}")

print("\nSPACER THICKNESSES (SiO2):")
for i,h in enumerate(hsp_list):
    print(f"  hsp[{i}] = {h:.4f}")

print("\nFIXED PATTERN:")
print(FIXED_PATTERN.reshape(3,3))

# =========================================================
# FINAL DIAGNOSTIC SWEEP
# =========================================================
lams = np.linspace(1.38, 1.52, 600)

phis = []
T   = []
R   = []

for lam in lams:
    r, t = solve_rt(lam, h_pattern, hs_list, hsp_list, hs_dbr, a)
    if r is None:
        phis.append(np.nan)
        T.append(np.nan)
        R.append(np.nan)
    else:
        phis.append(np.angle(r))
        T.append(t)
        R.append(abs(r)**2)

phis = np.unwrap(phis)

phi = np.array(phis)
T   = np.array(T)
R   = np.array(R)

# =========================================================
# CHECK PROFESSOR'S SUCCESS CONDITION
# =========================================================
T_peaks = []

for lam0 in lambda_targets:
    idx = np.argmin(np.abs(lams - lam0))
    T_peaks.append(T[idx])

T_peaks = np.array(T_peaks)

print("\nTRANSMISSION AT TARGET PEAKS:")
for i,(lam0,val) in enumerate(zip(lambda_targets, T_peaks)):
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
ax[0].set_title("Final Stage-2 Response")

ax[1].plot(lams, T, lw=2)
ax[1].set_ylabel("Transmission")

ax[2].plot(lams, R, lw=2)
ax[2].set_ylabel("Reflection")
ax[2].set_xlabel("Wavelength (µm)")

for lam in lambda_targets:
    for axt in ax:
        axt.axvline(lam, color='r', ls='--', alpha=0.35)

for axt in ax:
    axt.grid(True)

plt.tight_layout()
plt.show()