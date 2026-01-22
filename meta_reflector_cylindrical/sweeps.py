import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *

# ---------------------------------------------------------
# Material model: epsilon(lambda)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Cylinder epsilon grid
# ---------------------------------------------------------
def get_epgrids_cylinder(r, et):
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')

    x_c = x - L1[0] / 2
    y_c = y - L2[1] / 2

    mask = (x_c**2 + y_c**2) < r**2

    epgrid = np.ones((Nx, Ny), dtype=complex) * 1.0
    epgrid[mask] = et

    return epgrid


# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
def solver_system(f):

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

    epgrid = get_epgrids_cylinder(r, es).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    # locate (0,0) diffraction order
    # index of (0,0) diffraction order
    k0 = np.where((obj.G[:,0] == 0) & (obj.G[:,1] == 0))[0][0]
    nV = obj.nG

    # pick the polarization that is actually excited
    if np.abs(ai[k0]) > np.abs(ai[k0 + nV]):
        a00 = ai[k0]
        b00 = bi[k0]
    else:
        a00 = ai[k0 + nV]
        b00 = bi[k0 + nV]

    r00 = b00 / a00
    

    return r00, Ti[k0]


# ---------------------------------------------------------
# Wavelength sweep
# ---------------------------------------------------------
wavelengths = np.linspace(1.2, 1.27, 40)   # microns
phases = []
reflectance = []

for wl in wavelengths:
    f = 1 / wl
    r00, R00 = solver_system(f)
    phases.append(np.angle(r00))
    reflectance.append(R00)

phases = np.unwrap(np.array(phases))
reflectance = np.array(reflectance)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
#plt.plot(wavelengths, phases, lw=2)
plt.plot(wavelengths, reflectance, 'r--', lw=2)

plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Phase vs wavelength")

plt.text(
    0.02, 0.05,
    f"Radius r = {r:.3f} µm\nHeight h = {h:.3f} µm",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.grid(True)
plt.tight_layout()
plt.show()
