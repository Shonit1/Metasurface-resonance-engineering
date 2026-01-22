import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

"""
STAGE-1:
Find a single geometry that exhibits
large linear phase slopes in TWO SEPARATE wavelength windows.

This is a pole-proximity hunter, NOT a pole enforcer.
"""

# =========================================================
# WAVELENGTH WINDOWS
# =========================================================
lams1 = np.linspace(1.45, 1.50, 12)   # Window 1
lams2 = np.linspace(2, 2.05, 12)   # Window 2

# =========================================================
# MATERIAL MODEL ε(λ)
# =========================================================
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

# =========================================================
# CYLINDER ε-GRID
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')
    x_c = x - a/2
    y_c = y - a/2
    mask = (x_c**2 + y_c**2) < r**2
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair
    epgrid[mask] = et
    return epgrid

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1 = [a, 0]
    L2 = [0, a]
    es = epsilon_lambda(1/f)

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)

    for _ in range(5):
        obj.Add_LayerUniform(hs, es)
        obj.Add_LayerUniform(hsio2, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    epgrid = get_epgrids_cylinder(r, es, a).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0+nV]):
        return bi[k0]/ai[k0]
    else:
        return bi[k0+nV]/ai[k0+nV]

# =========================================================
# PHASE COMPUTATION (GENERIC)
# =========================================================
def compute_phase(lams, params):
    r, h, hsio2, a = params
    phis = []
    for lam in lams:
        r00 = solver_system(1/lam, r, h, hsio2, a)
        phis.append(np.angle(r00))
    return np.unwrap(np.array(phis))

# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array([
    [0.05, 0.35],   # r
    [0.05, 0.50],   # h
    [0.20, 0.65],   # hsio2
    [0.30, 0.80]    # a
])

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# LOSS FUNCTION: TWO-WINDOW SLOPE MAXIMIZATION
# =========================================================
def loss_two_slopes(x,
                    w1=1.0,
                    w2=1.0,
                    alpha=0,
                    beta=0.5):

    params = decode(x)

    # --- Window 1 ---
    phi1 = compute_phase(lams1, params)
    A1, B1 = np.polyfit(lams1, phi1, 1)
    rms1 = np.sqrt(np.mean((phi1 - (A1*lams1 + B1))**2))

    # --- Window 2 ---
    phi2 = compute_phase(lams2, params)
    A2, B2 = np.polyfit(lams2, phi2, 1)
    rms2 = np.sqrt(np.mean((phi2 - (A2*lams2 + B2))**2))

    # --- Anti-cheating: slopes must be comparable ---
    imbalance_penalty = (abs(A1) - abs(A2))**2

    loss = (
        - (w1*abs(A1) + w2*abs(A2))
        + alpha*(rms1 + rms2)
        + beta*imbalance_penalty
    )

    print(
        f"A1={A1:.2f}, A2={A2:.2f}, "
        f"rms1={rms1:.2e}, rms2={rms2:.2e}, "
        f"LOSS={loss:.2f}"
    )

    return loss

# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array([0.20, 0.20, 0.40, 0.55])
x0 = (x0_phys - bounds[:,0])/(bounds[:,1]-bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0,
    0.15,
    {
        "popsize": 10,
        "maxiter": 50,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_two_slopes(x) for x in xs]
    es.tell(xs, losses)

best_x = es.result.xbest
best_params = decode(best_x)

print("\n===== TWO-SLOPE OPTIMUM =====")
print("r     =", best_params[0])
print("h     =", best_params[1])
print("hsio2 =", best_params[2])
print("a     =", best_params[3])

# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phi1 = compute_phase(lams1, best_params)
phi2 = compute_phase(lams2, best_params)

A1,_ = np.polyfit(lams1, phi1, 1)
A2,_ = np.polyfit(lams2, phi2, 1)

plt.figure(figsize=(6,4))
plt.plot(lams1, phi1, 'o-', label=f'Window 1 (A1={A1:.1f})')
plt.plot(lams2, phi2, 'o-', label=f'Window 2 (A2={A2:.1f})')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("Two-window slope maximization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
