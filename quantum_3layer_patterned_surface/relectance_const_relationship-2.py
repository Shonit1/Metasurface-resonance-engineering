import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *


'This finds the resonance at two different wavelengths'




# =========================================================
# WAVELENGTH BANDS
# =========================================================
lambdas_A1 = np.linspace(0.73, 0.77, 6)
lambdas_A2 = np.linspace(1.48, 1.52, 6)



n_sio2 = 1.45
lambda_dbr = 1.50
hsio2_dbr = lambda_dbr / (4 * n_sio2)   # ≈ 0.259 µm

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

    if wavelength < 1.0:
        n = 3.7
    else:
        n = _cache["li"](wavelength)

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
                mask = (
                    (X>=i*dx)&(X<(i+1)*dx)&
                    (Y>=j*dy)&(Y<(j+1)*dy)
                )
                ep[mask] = eps
    return ep


# =========================================================
# RCWA SOLVER (CORRECT PHASE)
# =========================================================
def solver_system(f, p1, p2, h1, h2, hs, hsio2_spacer, a):

    eps_si = epsilon_lambda(1/f)
    obj = grcwa.obj(nG, [a,0], [0,a], f, theta, phi, verbose=0)

    # Top region
    obj.Add_LayerUniform(0.1, eair)

    # Patterned layer 1
    obj.Add_LayerGrid(h1, Nx, Ny)

    # 🔴 SiO2 spacer (OPTIMIZED)
    obj.Add_LayerUniform(hsio2_spacer, esio2)

    # Patterned layer 2
    obj.Add_LayerGrid(h2, Nx, Ny)

    # DBR stack (FIXED)
    for _ in range(5):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsio2_dbr, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    ep1 = get_epgrid_3x3(p1, eps_si, a).flatten()
    ep2 = get_epgrid_3x3(p2, eps_si, a).flatten()
    obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

    obj.MakeExcitationPlanewave(p_amp=1, p_phase=0, s_amp=0, s_phase=0)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0+nV]):
        r00 = bi[k0] / ai[k0]
    else:
        r00 = bi[k0+nV] / ai[k0+nV]

    return r00, Ri[k0], Ti[k0]


# =========================================================
# BAND RESPONSE
# =========================================================
def compute_band(params, lambdas):
    p1, p2 = params[:9], params[9:18]
    h1, h2, hs, hsio2_spacer, a = params[18:]

    phi, R, T = [], [], []

    for lam in lambdas:
        r00, R00, T00 = solver_system(
            1/lam, p1, p2, h1, h2, hs, hsio2_spacer, a
        )
        phi.append(np.angle(r00))
        R.append(R00)
        T.append(T00)

    return np.unwrap(phi), np.array(R), np.array(T)



# =========================================================
# BOUNDS
# =========================================================
bounds = np.array(
    [[0,1]]*18 +
    [
        [0.05,0.35],   # h1
        [0.05,0.35],   # h2
        [0.20,0.28],   # hs (Si in DBR)
        [0.05,0.40],   # hsio2_spacer ✅
        [0.10,0.75]    # a
    ]
)


def decode(x):
    x = np.clip(x,0,1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])


# =========================================================
# TOP 10 STORAGE
# =========================================================
TOP_K = 10
top = []

def update_top(loss, params):
    global top
    top.append((loss, params.copy()))
    top = sorted(top, key=lambda x: x[0])[:TOP_K]


def print_geometry(i, loss, params):
    p1 = (params[:9] > 0.5).reshape(3,3).astype(int)
    p2 = (params[9:18] > 0.5).reshape(3,3).astype(int)
    h1, h2, hs, hsio2_spacer, a = params[18:]

    print("\n" + "="*50)
    print(f"GEOMETRY #{i}")
    print(f"LOSS = {loss:.3e}")

    print("\nPattern layer 1:")
    print(p1)

    print("\nPattern layer 2:")
    print(p2)

    print("\nContinuous parameters:")
    print(f"h1              = {h1:.4f}")
    print(f"h2              = {h2:.4f}")
    print(f"hs (Si DBR)     = {hs:.4f}")
    print(f"hsio2_spacer    = {hsio2_spacer:.4f}")
    print(f"hsio2_dbr FIXED = {hsio2_dbr:.4f}")
    print(f"a               = {a:.4f}")



# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_function(x):

    params = decode(x)

    phi1, R1, T1 = compute_band(params, lambdas_A1)
    phi2, R2, T2 = compute_band(params, lambdas_A2)

    A1,_ = np.polyfit(lambdas_A1, phi1, 1)
    A2,_ = np.polyfit(lambdas_A2, phi2, 1)

    T075 = T1[np.argmin(abs(lambdas_A1-0.75))]
    T150 = T2[np.argmin(abs(lambdas_A2-1.50))]

    penalty_T = max(0,0.8-T075)**2 + max(0,0.8-T150)**2

    loss = -(A1**2 + A2**2) + 1e5*penalty_T

    print(
        f"A1={A1:.2e}, A2={A2:.2e}, "
        f"T(0.75)={T075:.2f}, T(1.50)={T150:.2f}, "
        f"LOSS={loss:.2e}"
    )

    update_top(loss, params)
    return loss


# =========================================================
# CMA
# =========================================================
x0 = np.full(len(bounds), 0.5)
es = cma.CMAEvolutionStrategy(x0, 0.3, {"popsize": 10, "maxiter": 40})

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_function(x) for x in xs])


# =========================================================
# FINAL PLOTS (TOP 10)
# =========================================================
for i,(loss,params) in enumerate(top):

    # 🔹 PRINT GEOMETRY
    print_geometry(i, loss, params)

    # 🔹 COMPUTE RESPONSE
    phi1,R1,T1 = compute_band(params, lambdas_A1)
    phi2,R2,T2 = compute_band(params, lambdas_A2)

    # 🔹 PLOTS
    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.plot(lambdas_A1, phi1, 'o-', label="A1")
    plt.plot(lambdas_A2, phi2, 'o-', label="A2")
    plt.title(f"Phase #{i}")
    plt.legend()

    plt.subplot(132)
    plt.plot(lambdas_A1, R1, 'o-', label="A1")
    plt.plot(lambdas_A2, R2, 'o-', label="A2")
    plt.title("Reflectance")
    plt.legend()

    plt.subplot(133)
    plt.plot(lambdas_A1, T1, 'o-', label="A1")
    plt.plot(lambdas_A2, T2, 'o-', label="A2")
    plt.title("Transmittance")
    plt.legend()

    plt.show()

