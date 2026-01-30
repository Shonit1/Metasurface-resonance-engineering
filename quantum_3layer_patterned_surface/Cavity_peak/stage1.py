import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# WAVELENGTH BAND (SHARP PEAK BIAS @ 1.5 µm)
# =========================================================
lambdas_res = np.linspace(1.49, 1.51, 7)
dx = lambdas_res[1] - lambdas_res[0]
c = len(lambdas_res) // 2   # center index

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

    dxg = a / 3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[
                    (X>=i*dxg)&(X<(i+1)*dxg)&
                    (Y>=j*dxg)&(Y<(j+1)*dxg)
                ] = eps
    return ep


# =========================================================
# RCWA SOLVER (FIXED DBR SiO2)
# =========================================================
def solver_system(f, p1, p2, h1, h2, hs, hsio2_spacer, a):
    try:
        
        eps_si = epsilon_lambda(1/f)
        obj = grcwa.obj(nG, [a,0], [0,a], f, theta, phi, verbose=0)

        # Air
        obj.Add_LayerUniform(0.1, eair)

        # Patterned layer 1
        obj.Add_LayerGrid(h1, Nx, Ny)

        # ✅ SiO2 SPACER (OPTIMIZED)
        obj.Add_LayerUniform(hsio2_spacer, esio2)

        # Patterned layer 2
        obj.Add_LayerGrid(h2, Nx, Ny)

        # ✅ DBR STACK (FIXED SiO2 THICKNESS)
        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        # Air
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
def compute_band(params):
    p1, p2 = params[:9], params[9:18]
    h1, h2, hs, hsio2_spacer, a = params[18:]

    T = []
    for lam in lambdas_res:
        val = solver_system(
            1/lam, p1, p2, h1, h2, hs, hsio2_spacer, a
        )
        if val is None or np.isnan(val):
            return None
        T.append(val)

    return np.array(T)


# =========================================================
# SAFE BOUNDS (NO PATHOLOGIES)
# =========================================================
bounds = np.array(
    [[0,1]]*18 +
    [
        [0.05,0.50],   # h1
        [0.05,0.50],   # h2
        [0.2,0.27],   # hs (Si in DBR)
        [0.05,0.30],   # hsio2_spacer ONLY
        [0.10,0.70]    # a
    ]
)

def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])


# =========================================================
# TOP-10 STORAGE (STABLE ONLY)
# =========================================================
TOP_K = 10
top = []

def update_top(loss, params):
    if compute_band(params) is None:
        return
    global top
    top.append((loss, params.copy()))
    top = sorted(top, key=lambda x: x[0])[:TOP_K]


# =========================================================
# LOSS FUNCTION (SHARP PEAK ENFORCEMENT)
# =========================================================
def loss_function(x):

    params = decode(x)
    T = compute_band(params)

    if T is None or len(T) < 5:
        return 1e9

    c = len(T) // 2
    Tc = T[c]

    # -------------------------------------------------
    # 1️⃣ Peak existence (center must be a local max)
    # -------------------------------------------------
    peak_penalty = max(0, T[c-1] - Tc) + max(0, T[c+1] - Tc)

    # -------------------------------------------------
    # 2️⃣ Edge suppression (broad background rejection)
    # -------------------------------------------------
    edge_penalty = np.sum(np.maximum(0, T[[0, -1]] - Tc))

    # -------------------------------------------------
    # 3️⃣ Coarse curvature sign (ONLY sign, not magnitude)
    # -------------------------------------------------
    dx = lambdas_res[1] - lambdas_res[0]
    d2C = (T[c+1] - 2*Tc + T[c-1]) / (dx*dx)

    curvature_penalty = max(0, d2C)   # want concave peak

    # -------------------------------------------------
    # 4️⃣ Transmission floor (avoid trivial solutions)
    # -------------------------------------------------
    transmission_penalty = max(0, 0.3 - Tc)

    # -------------------------------------------------
    # 5️⃣ Geometry sanity (avoid pathological thin layers)
    # -------------------------------------------------
    h1, h2, _, hsio2_spacer, a = params[18:]
    thin_penalty = 0.0
    for v, vmin in [(h1,0.12),(h2,0.12),(hsio2_spacer,0.12),(a,0.30)]:
        if v < vmin:
            thin_penalty += (vmin - v)**2

    # -------------------------------------------------
    # FINAL LOSS (carefully scaled)
    # -------------------------------------------------
    loss = (
        1e4 * peak_penalty +
        1e3 * edge_penalty +
        1e3 * curvature_penalty +
        1e3 * transmission_penalty +
        1e4 * thin_penalty
    )

    print(
        f"Tc={Tc:.3f}, "
        f"peak_pen={peak_penalty:.2e}, "
        f"edge_pen={edge_penalty:.2e}, "
        f"d2C={d2C:.2e}, "
        f"LOSS={loss:.2e}"
    )

    update_top(loss, params)
    return loss





# =========================================================
# CMA
# =========================================================
x0 = np.full(len(bounds), 0.5)
es = cma.CMAEvolutionStrategy(x0, 0.25, {"popsize": 10, "maxiter": 40})

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])


# =========================================================
# FINAL REPLAY (TOP-10)
# =========================================================
for i,(loss,params) in enumerate(top):
    print("\n" + "="*60)
    print(f"STAGE-1 | GEOMETRY #{i} | LOSS={loss:.3e}")

    T = compute_band(params)
    if T is None:
        print("⚠️ Failed replay")
        continue

    plt.plot(lambdas_res, T, 'o-')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transmission")
    plt.title(f"Sharp-peak biased spectrum #{i}")
    plt.grid(True)
    plt.show()
