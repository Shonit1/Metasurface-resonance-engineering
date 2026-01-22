import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import grcwa
from config import *

# =========================================================
# Wavelength range (µm)
# =========================================================
lambdas = np.linspace(1.45, 1.50, 25)

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
# Cylinder ε-grid
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')
    x_c = x - a / 2
    y_c = y - a / 2
    mask = (x_c**2 + y_c**2) < r**2
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair
    epgrid[mask] = et
    return epgrid


# =========================================================
# RCWA solver
# =========================================================
def solver_system(f, r, h, hsio2, a):

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

    epgrid = get_epgrids_cylinder(r, es, a).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(p_amp=1, p_phase=0,
                               s_amp=0, s_phase=0, order=0)

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
    r, h, hsio2, a = params
    phis = []

    for lam in lambdas:
        f = 1 / lam
        r00 = solver_system(f, r, h, hsio2, a)
        phis.append(np.angle(r00))

    return np.unwrap(np.array(phis))


# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_function(params, alpha=10.0):

    phis = compute_phase(params)

    # Linear fit: phi = A λ + B
    A, B = np.polyfit(lambdas, phis, 1)

    phi_fit = A * lambdas + B
    residuals = phis - phi_fit

    rms_error = np.sqrt(np.mean(residuals**2))

    # Maximize |A|, penalize nonlinearity
    loss = -np.abs(A) + alpha * rms_error

    print(f"A = {A:.3f}, RMS = {rms_error:.4e}, LOSS = {loss:.3f}")

    return loss


# =========================================================
# INITIAL GUESS (your current design)
# =========================================================
x0 = np.array([
    0.233,  # r
    0.120,  # h
    0.300,  # hsio2
    0.600   # a
])

# =========================================================
# BOUNDS (VERY IMPORTANT)
# =========================================================
bounds = [
    (0.05, 0.45),   # r
    (0.05, 0.40),   # h
    (0.05, 0.60),   # hsio2
    (0.45, 0.80)    # a
]

# =========================================================
# OPTIMIZATION
# =========================================================
result = minimize(
    loss_function,
    x0,
    method="Nelder-Mead",
    bounds=bounds,
    options={"maxiter": 50, "disp": True}
)

r_opt, h_opt, hsio2_opt, a_opt = result.x

print("\n===== OPTIMIZED PARAMETERS =====")
print("r =", r_opt)
print("h =", h_opt)
print("hsio2 =", hsio2_opt)
print("a =", a_opt)


# =========================================================
# FINAL DIAGNOSTICS
# =========================================================
phis = compute_phase(result.x)
A_opt, B_opt = np.polyfit(lambdas, phis, 1)
phi_fit = A_opt * lambdas + B_opt

print("\n===== FINAL LINEAR FIT =====")
print("A (rad/µm) =", A_opt)
print("B (rad)    =", B_opt)
print("RMS error  =", np.std(phis - phi_fit))


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")
plt.plot(lambdas, phi_fit, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Unwrapped phase (rad)")
plt.title("Optimized linear phase vs wavelength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis - phi_fit, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual phase (rad)")
plt.title("Deviation from linearity")
plt.grid(True)
plt.tight_layout()
plt.show()
