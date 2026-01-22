import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import grcwa

# ============================================================
# -------------------- GLOBAL SETTINGS -----------------------
# ============================================================

theta = 0
phi = 0

nG = 31
Nx = Ny = 120

eair = 1.0
esio2 = 1.44**2

hs = 0.1
num_pairs = 1

# ============================================================
# ---------------- MATERIAL INTERPOLATION --------------------
# ============================================================

def epsilon_lambda(lam, _cache={}):
    if "interp" not in _cache:
        data = pd.read_csv(
            "C:\\Users\\ASUS\\Downloads\\Li-293K.csv"
        )
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values
        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["interp"](lam)**2

# ============================================================
# ---------------- GEOMETRY GRID -----------------------------
# ============================================================

def get_epgrid_cylinder(r, eps, a):
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xc = X - a/2
    Yc = Y - a/2
    mask = (Xc**2 + Yc**2) < r**2
    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps
    return ep.flatten()

# ============================================================
# ---------------- RCWA REFLECTION ---------------------------
# ============================================================

def reflection_coeff(lam, a, r=0.25, h=0.4, hsi=0.3):
    f = 1 / lam
    eps_si = epsilon_lambda(lam)

    L1 = [a, 0]
    L2 = [0, a]

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsi, esio2)

    for _ in range(num_pairs):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsi, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    epgrid = get_epgrid_cylinder(r, eps_si, a)
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(0, 0.0)
    nV = obj.nG
    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]

    # ---- p-polarized zeroth order ----
    a00 = ai[k0 + nV]
    b00 = bi[k0 + nV]

    return b00 / a00

# ============================================================
# ---------------- PARAMETER SWEEP ---------------------------
# ============================================================

lambdas = np.linspace(1.4, 1.8, 40)
a_vals  = np.linspace(0.8, 1.4, 40)

phase_map = np.zeros((len(a_vals), len(lambdas)))
R_map     = np.zeros_like(phase_map)

print("Running λ–a sweep...\n")

for i, a in enumerate(a_vals):
    for j, lam in enumerate(lambdas):
        try:
            r00 = reflection_coeff(lam, a)
            R_map[i, j] = np.abs(r00)**2
            phase_map[i, j] = np.angle(r00)
        except np.linalg.LinAlgError:
            R_map[i, j] = np.nan
            phase_map[i, j] = np.nan
# unwrap phase along λ
phase_map = np.unwrap(phase_map, axis=1)

# ============================================================
# ---------------------- PLOTTING ----------------------------
# ============================================================

plt.figure(figsize=(7,5))
plt.imshow(
    phase_map,
    extent=[lambdas[0], lambdas[-1], a_vals[0], a_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='twilight'
)
plt.colorbar(label="Reflection phase (rad)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Lattice constant a (µm)")
plt.title("Phase map: λ vs lattice constant a")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.imshow(
    R_map,
    extent=[lambdas[0], lambdas[-1], a_vals[0], a_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='inferno'
)
plt.colorbar(label="Reflectance |r|²")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Lattice constant a (µm)")
plt.title("Reflectance map: λ vs lattice constant a")
plt.tight_layout()
plt.show()
