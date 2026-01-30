import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# Wavelength range (µm)
# =========================================================
lambdas = np.linspace(1.5, 1.55, 20)

# =========================================================
# Material model ε(λ)
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
# 3×3 pixelated ε-grid
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    pattern = np.array(pattern).reshape(3, 3)
    pattern = (pattern > 0.5).astype(int)

    epgrid = np.ones((Nx, Ny), dtype=complex) * eair

    dx = a / 3
    dy = a / 3

    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i, j] == 1:
                mask = (
                    (X >= i * dx) & (X < (i + 1) * dx) &
                    (Y >= j * dy) & (Y < (j + 1) * dy)
                )
                epgrid[mask] = et

    return epgrid


# =========================================================
# RCWA solver
# =========================================================
def solver_system(f, pattern, h, hsio2, a):

    L1 = [a, 0]
    L2 = [0, a]
    es = epsilon_lambda(1 / f)

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)

    for _ in range(5):
        obj.Add_LayerUniform(hs, es)
        obj.Add_LayerUniform(hsio2, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    epgrid = get_epgrid_3x3(pattern, es, a).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


# =========================================================
# Phase computation
# =========================================================
def compute_phase(params):
    pattern = params[:9]
    h, hsio2, a = params[9:]

    phis = []
    for lam in lambdas:
        r00 = solver_system(1 / lam, pattern, h, hsio2, a)
        phis.append(np.angle(r00))

    return np.unwrap(np.array(phis))


# =========================================================
# PARAMETER BOUNDS
# =========================================================
bounds = np.array(
    [[0, 1]] * 9 +      # 9 pixel variables
    [
        [0.05, 0.40],  # h
        [0.05, 0.60],  # hsio2
        [0.45, 0.80]   # a
    ]
)

def decode(x):
    x = np.clip(x, 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_function(x, alpha=10.0, beta=0.0,gamma= 10):

    params = decode(x)
    phis = compute_phase(params)

    # Linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    d2phi = np.gradient(np.gradient(phis, lambdas), lambdas)
    curvature = np.mean(d2phi**2)

    loss = -gamma*A**2 + alpha * rms + beta * curvature

    print(f"A={A:.3f}, RMS={rms:.3e}, CURV={curvature:.3e}, LOSS={loss:.3f}")
    return loss


# =========================================================
# CMA INITIALIZATION
# =========================================================
x0_phys = np.array(
    [0.5] * 9 +      # initial pixel guess
    [0.12, 0.30, 0.60]
)

x0 = (x0_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
sigma0 = 0.20

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "popsize": 10,
        "maxiter": 40,
        "verb_disp": 1
    }
)

# =========================================================
# CMA LOOP
# =========================================================
while not es.stop():
    solutions = es.ask()
    losses = [loss_function(x) for x in solutions]
    es.tell(solutions, losses)

best_x = es.result.xbest
best_params = decode(best_x)

print("\n===== CMA OPTIMUM =====")
print("Pixel pattern (3×3):")
print((best_params[:9] > 0.5).reshape(3, 3).astype(int))
print("h     =", best_params[9])
print("hsio2 =", best_params[10])
print("a     =", best_params[11])


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(best_params)
A_opt, B_opt = np.polyfit(lambdas, phis, 1)
phi_fit = A_opt * lambdas + B_opt

print("\n===== FINAL LINEAR FIT =====")
print("A (rad/µm) =", A_opt)
print("B (rad)    =", B_opt)
print("RMS error  =", np.std(phis - phi_fit))


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6, 4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")
plt.plot(lambdas, phi_fit, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("CMA-optimized linear phase (3×3 geometry)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(lambdas, phis - phi_fit, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Deviation from linearity")
plt.grid(True)
plt.tight_layout()
plt.show()
