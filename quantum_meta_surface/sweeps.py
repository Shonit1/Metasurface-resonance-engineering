import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *

# =========================================================
# Wavelength sweep (µm)
# =========================================================
lambdas = np.linspace(1.5, 1.65, 200)

# =========================================================
# FIXED GEOMETRY (your optimized design)
# =========================================================
pattern = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
]).flatten()

h =  0.24539302211440123
hsio2 = 0.05
a = 0.7656775386521044

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
# RCWA solver (returns r00, t00)
# =========================================================
def solve_rt(f):

    es = epsilon_lambda(1 / f)

    L1 = [a, 0]
    L2 = [0, a]

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    # Superstrate
    obj.Add_LayerUniform(0.1, eair)

    # Patterned silicon layer
    obj.Add_LayerGrid(h, Nx, Ny)

    # Substrate stack
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

    # Get amplitudes
    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
    

    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    # Reflection
    if abs(ai[k0]) > abs(ai[k0 + nV]):
        r00 = bi[k0] / ai[k0]
    else:
        r00 = bi[k0 + nV] / ai[k0 + nV]

    
    Ri,Ti = obj.RT_Solve(normalize=1,byorder=1)
    return r00, Ri[0],Ti[0]


# =========================================================
# Spectral sweep
# =========================================================
R = []
T = []
phase = []

for lam in lambdas:
    r00, Ri,Ti = solve_rt(1 / lam)

    R.append(Ri)
    T.append(Ti)
    phase.append(np.angle(r00))

phase = np.unwrap(np.array(phase))
R = np.array(R)
T = np.array(T)


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6, 4))
plt.plot(lambdas, phase, 'o-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection phase (rad)")
plt.title("Reflection phase vs wavelength")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(lambdas, R, 'r-', label="Reflection |R|²")
plt.plot(lambdas, T, 'b--', label="Transmission |T|²")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Power")
plt.title("Reflection / Transmission spectra")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# Energy check (optional diagnostic)
# =========================================================
plt.figure(figsize=(6, 4))
plt.plot(lambdas, R + T, 'k-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("R + T")
plt.title("Energy conservation check")
plt.grid(True)
plt.tight_layout()
plt.show()
