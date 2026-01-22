import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import nG, theta, Nx, Ny, esio2, eair, hs
# =========================================================
# RCWA / SIMULATION PARAMETERS (from your config)
# =========================================================




        # SiO2 permittivity (adjust if dispersive)
               # silicon spacer thickness (µm)

# =========================================================
# MATERIAL MODEL ε(λ) FOR Si
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
def get_epgrids_cylinder(r, eps_si, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing="ij")

    x_c = x - a / 2
    y_c = y - a / 2

    mask = (x_c**2 + y_c**2) < r**2

    epgrid = np.ones((Nx, Ny), dtype=complex) * eair
    epgrid[mask] = eps_si

    return epgrid

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1 = [a, 0]
    L2 = [0, a]
    phi_inc = 0
    eps_si = epsilon_lambda(1 / f)

    obj = grcwa.obj(
        nG, L1, L2,
        f, theta, phi_inc,
        verbose=0
    )

    # Superstrate
    obj.Add_LayerUniform(0.1, eair)

    # Patterned layer
    obj.Add_LayerGrid(h, Nx, Ny)

    # Spacer
    obj.Add_LayerUniform(hsio2, esio2)

    # Multilayer stack
    for _ in range(5):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsio2, esio2)

    # Substrate
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    epgrid = get_epgrids_cylinder(r, eps_si, a).flatten()
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
# FIXED GEOMETRY (CRITICAL-COUPLING ONE)
# =========================================================
r     = 0.3035082512261301
h     = 0.0745696719591017
hsio2 = 0.4613939870153822
a     = 0.7206415591170998

# =========================================================
# WAVELENGTH SCAN
# =========================================================
lambdas = np.linspace(1.45, 1.60, 200)

R = []
phi = []

for lam in lambdas:
    r00 = solver_system(1 / lam, r, h, hsio2, a)
    R.append(np.abs(r00)**2)
    phi.append(np.angle(r00))

R = np.array(R)
phi = np.unwrap(np.array(phi))

# =========================================================
# PHASE DERIVATIVE
# =========================================================
dphi = np.gradient(phi, lambdas)

# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(lambdas, R, lw=2)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflectance |r|²")
plt.title("Reflectance spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, phi, lw=2)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection phase (rad)")
plt.title("Reflection phase spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(lambdas, np.abs(dphi), lw=2)
plt.xlabel("Wavelength (µm)")
plt.ylabel("|dφ/dλ| (rad/µm)")
plt.title("Phase dispersion")
plt.grid(True)
plt.tight_layout()
plt.show()
