import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# GLOBAL SETTINGS
# =========================================================

theta = 0.0
phi   = 0.0
DBR_PAIRS = 5

eair  = 1.0
esio2 = 1.44**2
hsio2_dbr = 0.2586

# =========================================================
# TARGET RESONANCES
# =========================================================
lambda_targets = np.array([
    1.40, 1.415, 1.430, 1.445, 1.460, 1.475, 1.490
])
dlam = 0.002

# =========================================================
# FIXED GEOMETRY (FROM STAGE-2)
# =========================================================
h_pattern = 0.2000
a_fixed   = 1.3451
hs_dbr    = 0.2200

hsp_list = np.array([0.1500]*7)

FIXED_PATTERN = np.array([
    [0.57213402, 0.49403910, 0.39748805],
    [0.23742280, 0.00000000, 0.00000000],
    [1.00000000, 0.00000000, 0.94079933]
])

N_CAV = 7

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
# 3×3 METASURFACE GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
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
# RCWA SOLVER
# =========================================================
def solve_rt(lam, hs_list):

    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG,[a_fixed,0],[0,a_fixed],1/lam,theta,phi,verbose=0)

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

        ep = get_epgrid_3x3(FIXED_PATTERN, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        r = complex(r)

        _, T = obj.RT_Solve(normalize=1, byorder=1)
        return r, T[k0]

    except Exception:
        return None, None

# =========================================================
# STAGE-2.5 OPTIMIZATION VARIABLES (ONLY hs_i)
# =========================================================
N_PARAMS = N_CAV
bounds = np.array([[0.20, 0.30]] * N_CAV)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# STAGE-2.5 LOSS FUNCTION
# =========================================================
def stage25_loss(x):

    hs_list = decode(x)

    curvatures = []
    peaks = []

    for lam0 in lambda_targets:

        phis = []
        Ts   = []

        for lam in (lam0-dlam, lam0, lam0+dlam):
            r, T = solve_rt(lam, hs_list)
            if r is None:
                return 1e6
            phis.append(np.angle(r))
            Ts.append(T)

        phis = np.unwrap(phis)
        d2phi = (phis[2] - 2*phis[1] + phis[0]) / dlam**2

        curvatures.append(abs(d2phi))
        peaks.append(Ts[1])

    curvatures = np.array(curvatures)
    peaks      = np.array(peaks)

    # Force all resonances to exist
    curvature_loss = np.sum(1.0 / (curvatures + 1e-6))

    # Prevent collapse to zero transmission
    T_floor = 0.15
    transmission_guard = np.sum(np.maximum(0, T_floor - peaks)**2)

    loss = curvature_loss + 0.5 * transmission_guard

    print(
        f"min|φ''|={curvatures.min():.2e}, "
        f"Tmin={peaks.min():.3f}, "
        f"loss={loss:.3e}"
    )

    return loss

# =========================================================
# CMA-ES (STAGE-2.5)
# =========================================================
x0 = np.full(N_PARAMS, 0.5)

es = cma.CMAEvolutionStrategy(
    x0,
    0.08,
    {"popsize":10, "maxiter":60}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [stage25_loss(x) for x in xs])

# =========================================================
# EXTRACT BEST SOLUTION
# =========================================================
best_hs = decode(es.result.xbest)

print("\n" + "="*60)
print("STAGE-2.5 COMPLETE — OPTIMIZED CAVITY THICKNESSES")
print("="*60)

for i,h in enumerate(best_hs):
    print(f"hs[{i}] = {h:.4f}")

# =========================================================
# FINAL DIAGNOSTIC SWEEP
# =========================================================
lams = np.linspace(1.38,1.52,600)
phis = []
T = []

for lam in lams:
    r,t = solve_rt(lam, best_hs)
    phis.append(np.angle(r) if r is not None else np.nan)
    T.append(t if t is not None else np.nan)

phis = np.unwrap(phis)

plt.figure(figsize=(8,6))
plt.plot(lams, phis, lw=2)
for lam0 in lambda_targets:
    plt.axvline(lam0, ls="--", color="r", alpha=0.4)
plt.ylabel("Reflection phase (rad)")
plt.xlabel("Wavelength (µm)")
plt.title("Stage-2.5: Phase (All Resonances Visible)")
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(lams, T, lw=2)
for lam0 in lambda_targets:
    plt.axvline(lam0, ls="--", color="r", alpha=0.4)
plt.ylabel("Transmission")
plt.xlabel("Wavelength (µm)")
plt.title("Stage-2.5: Transmission")
plt.grid()
plt.show()
