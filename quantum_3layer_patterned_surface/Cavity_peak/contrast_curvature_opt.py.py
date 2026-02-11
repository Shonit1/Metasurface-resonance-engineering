import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# STAGE-1: SHARP RESONANCE SHAPE ENFORCEMENT
# =========================================================

# --- three small windows ---
lambdas_L = np.linspace(1.49990, 1.49995, 7)   # left shoulder
lambdas_C = np.linspace(1.49995, 1.50000, 7)   # center peak
lambdas_R = np.linspace(1.50000, 1.50005, 7)   # right shoulder

# =========================================================
# MATERIAL MODEL ε(λ)
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
# 3×3 GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = (np.array(pattern).reshape(3,3) > 0.5).astype(int)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[(X>=i*dx)&(X<(i+1)*dx)&
                   (Y>=j*dy)&(Y<(j+1)*dy)] = eps
    return ep


# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, p1, p2, h1, h2, hs, hsio2, a):
    try:
        eps_si = epsilon_lambda(1/f)
        obj = grcwa.obj(nG, [a,0], [0,a], f, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(p1, eps_si, a).flatten()
        ep2 = get_epgrid_3x3(p2, eps_si, a).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0)

        _, T = obj.RT_Solve(normalize=1, byorder=1)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]

        return T[k0]

    except Exception:
        return None


# =========================================================
# BAND RESPONSE
# =========================================================
def compute_band(params, lambdas):
    p1, p2 = params[:9], params[9:18]
    h1, h2, hs, hsio2, a = params[18:]

    T = []
    for lam in lambdas:
        t = solver_system(1/lam, p1, p2, h1, h2, hs, hsio2, a)
        if t is None or np.isnan(t):
            return None
        T.append(t)
    return np.array(T)


def curvature(T):
    c = len(T)//2
    return T[c+1] - 2*T[c] + T[c-1]


# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0,1]]*18 +
    [
        [0.08,0.35],   # h1
        [0.08,0.35],   # h2
        [0.20,0.28],   # hs
        [0.05,0.40],   # hsio2 spacer ONLY
        [0.90,1.80]    # a  (IMPORTANT)
    ]
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])


# =========================================================
# TOP-10 STORAGE (ROBUST)
# =========================================================
TOP_K = 10
top = []

def update_top(loss, params):
    if not np.isfinite(loss):
        return
    top.append((loss, params.copy()))
    top.sort(key=lambda x: x[0])
    del top[TOP_K:]


# =========================================================
# LOSS FUNCTION (FINAL STAGE-1)
# =========================================================
def loss_function(x):

    params = decode(x)

    TL = compute_band(params, lambdas_L)
    TC = compute_band(params, lambdas_C)
    TR = compute_band(params, lambdas_R)

    if TL is None or TC is None or TR is None:
        return 1e6

    c = len(TC)//2
    Tc  = TC[c]
    TLc = TL[c]
    TRc = TR[c]

    # --- curvature (dimensionless) ---
    d2L = curvature(TL)
    d2C = curvature(TC)
    d2R = curvature(TR)

    # --- must be a local max (soft) ---
    local_max_penalty = max(0.0, TLc - Tc)**2 + max(0.0, TRc - Tc)**2

    # --- sharpness ---
    sharpness = max(0.0, -d2C)
    if sharpness < 1e-5:
        return 0.5 + (1e-5 - sharpness)*1e4

    # --- contrast ---
    contrast = Tc - 0.5*(TLc + TRc)

    # --- transmission soft target ---
    T_target = 0.6
    T_penalty = max(0.0, T_target - Tc)**2

    loss = (
        + 5.0 * local_max_penalty
        + 2.0 * T_penalty
        - 2000.0 * sharpness
        - 20.0 * contrast
    )

    print(
        f"Tc={Tc:.3f}, contrast={contrast:.3f}, "
        f"sharp={sharpness:.2e}, LOSS={loss:.2e}"
    )

    update_top(loss, params)
    return loss



# =========================================================
# CMA OPTIMIZATION
# =========================================================
x0 = np.full(len(bounds), 0.5)

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize": 10, "maxiter": 40}
)

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])


# =========================================================
# FINAL REPLAY (TOP-10)
# =========================================================
for i, (loss, params) in enumerate(top):

    p1 = (params[:9] > 0.5).reshape(3,3).astype(int)
    p2 = (params[9:18] > 0.5).reshape(3,3).astype(int)
    h1, h2, hs, hsio2, a = params[18:]

    print("\n" + "="*60)
    print(f"STAGE-1 | GEOMETRY #{i}")
    print(f"LOSS = {loss:.3e}")
    print("Pattern 1:\n", p1)
    print("Pattern 2:\n", p2)
    print(f"h1={h1:.4f}, h2={h2:.4f}, hs={hs:.4f}, hsio2={hsio2:.4f}, a={a:.4f}")

    lambdas = np.concatenate([lambdas_L, lambdas_C, lambdas_R])
    T = np.concatenate([
        compute_band(params, lambdas_L),
        compute_band(params, lambdas_C),
        compute_band(params, lambdas_R)
    ])

    plt.plot(lambdas, T, 'o-')
    plt.axvline(1.5, color='r', linestyle='--')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transmission")
    plt.title(f"Stage-1 Resonance #{i}")
    plt.grid(True)
    plt.show()
