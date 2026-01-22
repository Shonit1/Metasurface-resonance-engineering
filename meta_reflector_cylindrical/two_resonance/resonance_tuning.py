import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# KNOWN FIRST POLE (LOCKED)
# =========================================================
lambda1 = 1.474       # µm
win1 = 0.020          # locking window

# SECOND RESONANCE (TO PUSH)
lambda2_guess = 1.24  # from sweep
win2 = 0.030

lams1 = np.linspace(lambda1 - win1, lambda1 + win1, 18)
lams2 = np.linspace(lambda2_guess - win2, lambda2_guess + win2, 20)

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
    return _cache["interp"](wavelength)**2

# =========================================================
# CYLINDER ε GRID
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')
    x_c = x - a/2
    y_c = y - a/2
    mask = (x_c**2 + y_c**2) < r**2
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = et
    return ep

# =========================================================
# RCWA SOLVER (REFLECTION + TRANSMISSION)
# =========================================================
def solver_RT(f, r, h, hsio2, a):

    L1, L2 = [a, 0], [0, a]
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

    ep = get_epgrids_cylinder(r, es, a).flatten()
    obj.GridLayer_geteps(ep)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )
    last_layer = obj.Layer_N - 1
    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    at, bt = obj.GetAmplitudes(which_layer=last_layer, z_offset=0.0)

    k0 = np.where((obj.G[:,0] == 0) & (obj.G[:,1] == 0))[0][0]
    nV = obj.nG


    R = bi[k0+nV] / ai[k0+nV]
    T = bt[k0+nV] / ai[k0+nV]

    return R, T

# =========================================================
# PHASE UTILITIES
# =========================================================
def unwrap_phase(z):
    return np.unwrap(np.angle(z))

def phase_derivatives(lams, z):
    phi = unwrap_phase(z)
    dphi = np.gradient(phi, lams)
    ddphi = np.gradient(dphi, lams)
    return phi, dphi, ddphi

# =========================================================
# PARAMETER BOUNDS (TIGHT AROUND YOUR GEOMETRY)
# =========================================================
bounds = np.array([
    [0.28, 0.33],    # r
    [0.06, 0.09],    # h
    [0.40, 0.55],    # hsio2
    [0.65, 0.78]     # a
])

def decode(x):
    x = np.clip(x, 0, 1)
    return bounds[:,0] + x * (bounds[:,1] - bounds[:,0])

# =========================================================
# LOSS FUNCTION: LOCK FIRST POLE, PUSH SECOND
# =========================================================
def loss_dual_pole(x,
                   w_lock=3.0,
                   w_push=4.0,
                   w_amp=0.4):

    r, h, hsio2, a = decode(x)

    # -------- LOCK FIRST POLE --------
    R1 = np.array([solver_RT(1/lam, r, h, hsio2, a)[0] for lam in lams1])
    phi1, dphi1, ddphi1 = phase_derivatives(lams1, R1)

    lock_term = (
        - np.max(np.abs(dphi1))
        - np.max(np.abs(ddphi1))
    )

    # -------- PUSH SECOND POLE --------
    R2, T2 = [], []
    for lam in lams2:
        R, T = solver_RT(1/lam, r, h, hsio2, a)
        R2.append(R)
        T2.append(T)

    R2 = np.array(R2)
    T2 = np.array(T2)

    phi2, dphi2, ddphi2 = phase_derivatives(lams2, R2)

    push_term = (
        - np.max(np.abs(dphi2))
        - np.max(np.abs(ddphi2))
    )

    # -------- TRANSMISSION SHARPNESS --------
    amp_term = np.var(np.abs(T2))

    loss = (
        w_lock * lock_term +
        w_push * push_term +
        w_amp  * amp_term
    )

    print(
        f"lock={lock_term:.2e}, "
        f"push={push_term:.2e}, "
        f"amp={amp_term:.2e}, "
        f"LOSS={loss:.2e}"
    )

    return loss

# =========================================================
# CMA INITIALIZATION (YOUR GEOMETRY)
# =========================================================
x0_phys = np.array([
    0.3035082512261301,
    0.0745696719591017,
    0.4613939870153822,
    0.7206415591170998
])

x0 = (x0_phys - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0,
    0.10,
    {"popsize": 10, "maxiter": 60}
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_dual_pole(x) for x in xs]
    es.tell(xs, losses)

best_params = decode(es.result.xbest)

print("\n===== OPTIMIZED GEOMETRY =====")
print("r     =", best_params[0])
print("h     =", best_params[1])
print("hsio2 =", best_params[2])
print("a     =", best_params[3])

# =========================================================
# FINAL CHARACTERIZATION
# =========================================================
lams = np.linspace(1.15, 1.65, 300)

R, T = [], []
for lam in lams:
    r0, t0 = solver_RT(1/lam, *best_params)
    R.append(r0)
    T.append(t0)

R = np.array(R)
T = np.array(T)

phi_R = unwrap_phase(R)
phi_T = unwrap_phase(T)

omega = 2*np.pi*3e8/(lams*1e-6)
tau_g = np.gradient(phi_R, omega)

# ----------------- PLOTS -----------------
plt.figure(figsize=(6,4))
plt.plot(lams, phi_R)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection phase (rad)")
plt.title("Phase winding")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lams, tau_g * 1e15)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Group delay (fs)")
plt.title("Group delay enhancement")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lams, phi_T)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmission phase (rad)")
plt.title("Transmission phase")
plt.grid(True)
plt.tight_layout()
plt.show()
