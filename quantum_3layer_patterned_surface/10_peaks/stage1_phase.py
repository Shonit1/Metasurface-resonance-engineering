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
N_CAV = 7                  # <-- OPTION B (10 peaks)
N_PEAKS = 7
DBR_PAIRS = 5

h_pattern = 0.3          # fixed in Stage-1
hsio2_dbr = 0.2586         # FIXED
dlam = 1e-4

lambda_targets = np.linspace(1.4, 1.5, N_PEAKS)

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


def safe_angle(r):
    """
    Robust phase extraction.
    Returns None if r is invalid.
    """
    try:
        r = complex(r)
        if not (np.isfinite(r.real) and np.isfinite(r.imag)):
            return None
        return np.angle(r)
    except Exception:
        return None


def save_stage1_geometry(params, slopes, ratio_min, ratio_std, tag="auto"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"stage1_SUCCESS_{tag}_{ts}.npz"

    np.savez(
        fname,
        params=params,
        slopes=slopes,
        ratio_min=ratio_min,
        ratio_std=ratio_std
    )

    print(f"\n💾 SAVED STAGE-1 GEOMETRY → {fname}\n")




# =========================================================
# RCWA SOLVER (COMPLEX r, SAFE)
# =========================================================
def solver_r(lam, pattern, hs_list, hsp_list, hs_dbr, a):
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

        ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r_raw = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        r = complex(r_raw)   # 🔒 FORCE COMPLEX

        if not (np.isfinite(r.real) and np.isfinite(r.imag)):
            return None, None
        return r

    except Exception:
        return None, None



def solver_T(lam, pattern, hs_list, hsp_list, hs_dbr, a):
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

        ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1,0,0,0)

        
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        R,T = obj.RT_Solve(normalize=1,byorder=1)

        
        return R[k0], T[k0]

    except Exception:
        return None, None



# =========================================================
# PARAMETERS
# =========================================================
N_PARAMS = 9 + N_CAV + N_CAV + 1 + 1 + 1

bounds = np.array(
    [[0,1]]*9 +
    [[0.22,0.28]]*N_CAV +
    [[0.15,0.35]]*N_CAV +
    [[0.20,0.28]] +        # hs_dbr
    [[1.30,1.36]] +        # a
    [[-np.pi,np.pi]]       # phi0
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# TOP-10 STORAGE
# =========================================================
TOP_K = 10
top = []

def update_top(loss, p):
    top.append((loss, p.copy()))
    top.sort(key=lambda z: z[0])
    del top[TOP_K:]

# =========================================================
# STAGE-1 LOSS (PARTICIPATION FIRST)
# =========================================================
def loss_function(x):

    p = decode(x)

    pattern = p[:9]
    hs_list = p[9:9+N_CAV]
    hsp_list = p[9+N_CAV:9+2*N_CAV]
    hs_dbr  = p[9+2*N_CAV]
    a       = p[9+2*N_CAV+1]
    phi0    = p[9+2*N_CAV+2]

    slopes = []

    for lam0 in lambda_targets:
        phis = []
        for lam in (lam0-dlam, lam0, lam0+dlam):
            r = solver_r(lam,pattern,hs_list,hsp_list,hs_dbr,a)
            if r is None or not np.isfinite(r):
                return 1e6
            phi = safe_angle(r)
            if phi is None:
                return 1e6   # or continue / skip
            phis.append(phi)

        phis = np.unwrap(phis) + phi0
        slopes.append(abs((phis[2]-phis[0])/(2*dlam)))

    slopes = np.array(slopes)

    μ = slopes.mean()
    σ = slopes.std()
    m = slopes.min()

    ratio_std = σ/(μ+1e-9)
    ratio_min = m/(μ+1e-9)
    # =====================================================
# AUTO-SAVE SUCCESSFUL GEOMETRY (FAIL-SAFE)
# =====================================================
    if ratio_min >= 0.5:
        save_stage1_geometry(
            params=p,
            slopes=slopes,
            ratio_min=ratio_min,
            ratio_std=ratio_std,
            tag="min0p5"
        )


    loss = (
        2.0*(1-ratio_min)**2 +
        1.0*(ratio_std)**2
    )

    update_top(loss, p)

    print(
        f"<|dφ/dλ|>={μ:.2e}, "
        f"σ/μ={ratio_std:.2f}, "
        f"min/μ={ratio_min:.2f}, "
        f"a={a:.3f}"
    )

    return loss

# =========================================================
# CMA-ES
# =========================================================
x0 = np.full(N_PARAMS,0.5)
es = cma.CMAEvolutionStrategy(
    x0,0.25,{"popsize":12,"maxiter":60}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs,[loss_function(x) for x in xs])

# =========================================================
# FINAL PLOTS (TOP-10)
# =========================================================
lams = np.linspace(1.38,1.52,400)

for i,(loss,p) in enumerate(top):

    pattern = p[:9]
    hs_list = p[9:9+N_CAV]
    hsp_list = p[9+N_CAV:9+2*N_CAV]
    hs_dbr  = p[9+2*N_CAV]
    a       = p[9+2*N_CAV+1]
    phi0    = p[9+2*N_CAV+2]

    phis,R,T = [],[],[]

    for lam in lams:
        r = solver_r(lam,pattern,hs_list,hsp_list,hs_dbr,a)
        phis.append(np.angle(r))
        R0,T0 = solver_T(lam,pattern,hs_list,hsp_list,hs_dbr,a)
        R.append(R0)
        T.append(T0)

    phis = np.unwrap(phis) + phi0

    fig,ax = plt.subplots(3,1,figsize=(7,8),sharex=True)
    ax[0].plot(lams,phis); ax[0].set_ylabel("Phase")
    ax[1].plot(lams,T);   ax[1].set_ylabel("T")
    ax[2].plot(lams,R);   ax[2].set_ylabel("R")
    for l in lambda_targets:
        for axt in ax: axt.axvline(l,color='r',ls='--',alpha=0.3)
    ax[2].set_xlabel("λ (µm)")
    plt.suptitle(f"TOP {i} | loss={loss:.2e}")
    plt.show()
