import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import grcwa
from config import *

'''Ultra-zoom BIC diagnostics (4×4 single-pattern geometry)'''

# =========================================================
# USER INPUT: GEOMETRY (BEST FOUND)
# =========================================================
h1    = 0.283495    # patterned layer thickness
hs    = 0.299057   # DBR Si thickness
hsio2 = 0.051142    # spacer + DBR SiO2 thickness

# =========================================================
# LATTICE & PATTERN (4×4)
# =========================================================
P = np.array([
    [1,0,1,0],
    [0,0,0,0],
    [0,0,0,0],
    [1,1,0,1]
]) 

a_fixed = 0.9

# =========================================================
# RESONANCE CENTER & ZOOM
# =========================================================
lambda0_um = 1.50000
dlam_um    = 3e-3
Npts       = 501

lams_um = np.linspace(
    lambda0_um - dlam_um,
    lambda0_um + dlam_um,
    Npts
)

lams_nm = lams_um * 1000.0

# =========================================================
# MATERIAL MODEL (Si)
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
    return _cache["li"](wavelength)**2

# =========================================================
# 4×4 PERMITTIVITY GRID
# =========================================================
def get_epgrid_4x4(pattern, eps, a):
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    dx = dy = a / 4

    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(4):
        for j in range(4):
            if pattern[i, j]:
                ep[(X >= i*dx) & (X < (i+1)*dx) &
                   (Y >= j*dy) & (Y < (j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVER (R, T)
# =========================================================
def solver_rt(lam):
    try:
        eps_si = epsilon_lambda(lam)

        obj = grcwa.obj(
            nG, [a_fixed,0], [0,a_fixed],
            1/lam, theta, phi, verbose=0
        )

        # Air
        obj.Add_LayerUniform(0.1, eair)

        # Patterned metasurface
        obj.Add_LayerGrid(h1, Nx, Ny)

        # SiO2 spacer
        obj.Add_LayerUniform(hsio2, esio2)

        # DBR
        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        # Glass substrate
        obj.Add_LayerUniform(1.0, esio2)

        obj.Init_Setup()

        ep = get_epgrid_4x4(P, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(ep)

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

        if obj.G.size == 0:
            return None, None, None, None

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]

        t = Ti[k0]

        return r, Ri, Ti, obj

    except Exception:
        return None, None, None, None

# =========================================================
# BUILD SOLVER (FIELD VISUALIZATION)
# =========================================================
def build_solver(lam):
    eps_si = epsilon_lambda(lam)

    obj = grcwa.obj(
        nG, [a_fixed,0], [0,a_fixed],
        1/lam, theta, phi, verbose=0
    )

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h1, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)

    for _ in range(5):
        obj.Add_LayerUniform(hs, eps_si)
        obj.Add_LayerUniform(hsio2_dbr, esio2)

    obj.Add_LayerUniform(1.0, esio2)

    obj.Init_Setup()

    ep = get_epgrid_4x4(P, eps_si, a_fixed).flatten()
    obj.GridLayer_geteps(ep)

    obj.MakeExcitationPlanewave(1,0,0,0)

    return obj

# =========================================================
# FIELD INTENSITY (XZ SLICE)
# =========================================================
def compute_xz_field_intensity(obj, which_layer, z_min, z_max, Nz, y_index=None):
    z_vals = np.linspace(z_min, z_max, Nz)
    I_xz = np.zeros((Nz, Nx))

    if y_index is None:
        y_index = Ny // 2

    for i, z in enumerate(z_vals):
        E, _ = obj.Solve_FieldOnGrid(which_layer=which_layer, z_offset=z)
        Ex, Ey, Ez = E
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        I_xz[i, :] = I[:, y_index]

    x = np.linspace(0, a_fixed, Nx)
    return x, z_vals, I_xz

# =========================================================
# RESONANCE FINDER
# =========================================================
def find_resonance_in_range(lams, Rvals):
    lams = np.array(lams)
    Rvals = np.array(Rvals)
    idx = np.argmin(Rvals)
    return lams[idx], Rvals[idx]


'''


# =========================================================
# RCWA SCAN (R, T, phase, energy conservation)
# =========================================================
Ri_list, Ti_list = [], []
phi_list, lams_valid = [], []
energy_sum, r_list = [], []

for lam_um, lam_nm in zip(lams_um, lams_nm):
    r, Ri, Ti, obj = solver_rt(lam_um)

    if r is None or Ri is None or Ti is None:
        continue

    Ri_list.append(Ri)
    Ti_list.append(Ti)
    phi_list.append(np.angle(r))
    lams_valid.append(lam_nm)
    energy_sum.append(np.sum(Ri) + np.sum(Ti))
    r_list.append(np.abs(r)**2)

Ri_arr = np.array(Ri_list)
Ti_arr = np.array(Ti_list)

R00 = Ri_arr[:, 0]
T00 = Ti_arr[:, 0]

# =========================================================
# PLOT DIFFRACTION ORDERS
# =========================================================
plt.figure(figsize=(14,8))
plt.plot(lams_valid, R00, label="R00")
plt.plot(lams_valid, T00, label="T00")
plt.plot(lams_valid, r_list, "--", label="|r|² (from amplitudes)")
plt.plot(lams_valid, energy_sum, "--", color="black", label="Σ(R+T)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.title("Zeroth-order Reflection & Transmission")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# RESONANCE FINDING
# =========================================================
lam_res, Rmin = find_resonance_in_range(lams_um, r_list)
print(f"Resonance at λ = {lam_res:.6f} µm")

# =========================================================
# BUILD SOLVER AT RESONANCE
# =========================================================
obj = build_solver(lam_res)

# =========================================================
# FIELD PLOTTING UTILITIES
# =========================================================
def plot_xz_intensity(x, z, I, title):
    plt.figure(figsize=(6,4))
    plt.pcolormesh(x, z, I, shading="auto", cmap="inferno")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar(label="|E|²")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================================================
# FULL STRUCTURE FIELD MAP
# =========================================================
layer_thicknesses = (
    [h1, hsio2] +           # pattern + spacer
    [hs, hsio2_dbr] * 5     # DBR
)

layer_names = (
    ["Si pattern", "SiO2 spacer"] +
    ["Si DBR", "SiO2 DBR"] * 5
)

def compute_full_xz_intensity(obj, layer_thicknesses, Nz_per_layer=80):
    I_all, z_all = [], []
    z_offset = 0.0
    layer_bounds = []

    for i, h in enumerate(layer_thicknesses):
        x, z_loc, I_loc = compute_xz_field_intensity(
            obj,
            which_layer=i+1,
            z_min=0,
            z_max=h,
            Nz=Nz_per_layer
        )

        z_shifted = z_loc + z_offset
        I_all.append(I_loc)
        z_all.append(z_shifted)

        z_offset += h
        layer_bounds.append(z_offset)

    return (
        x,
        np.concatenate(z_all),
        np.vstack(I_all),
        layer_bounds
    )

def plot_full_xz_intensity(x, z, I, layer_bounds, title, fname):
    plt.figure(figsize=(6,6))
    plt.pcolormesh(x, z, I, shading="auto", cmap="inferno")
    plt.colorbar(label="|E|²")

    for zb in layer_bounds:
        plt.axhline(zb, color="white", lw=0.6, alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

# =========================================================
# COMPUTE & SAVE FIELD MAP
# =========================================================
x, z, Ixz, layer_bounds = compute_full_xz_intensity(
    obj,
    layer_thicknesses,
    Nz_per_layer=80
)

plot_full_xz_intensity(
    x, z, Ixz,
    layer_bounds,
    title=f"Quasi-BIC Field | λ={lam_res:.6f} µm",
    fname="full_xz_field.png"
)

print("✓ Saved: full_xz_field.png")













'''






# =========================================================
# RCWA SCAN (4×4 single-pattern compatible)
# =========================================================
R_list, T_list, phi_list, lams_valid = [], [], [], []

for lam_um, lam_nm in zip(lams_um, lams_nm):
    r, Ri, Ti, obj = solver_rt(lam_um)

    if r is None or Ri is None or Ti is None:
        continue

    R_list.append(np.abs(r)**2)   # reflectance from amplitude
    T_list.append(np.real(Ti[0])) # T00
    phi_list.append(np.angle(r))
    lams_valid.append(lam_nm)

R = np.array(R_list)
T = np.array(T_list)
phi = np.unwrap(np.array(phi_list))
lams_valid = np.array(lams_valid)

# =========================================================
# PHASE DERIVATIVE (rad / nm)
# =========================================================
dphi_dlambda = np.gradient(phi, lams_valid)

# =========================================================
# PLOTTING
# =========================================================
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.plot(lams_valid, R)
plt.ylabel("Reflectance R")
plt.xlabel("Wavelength (nm)")
plt.title("Reflectance")
plt.grid(alpha=0.3)

plt.subplot(2,2,2)
plt.plot(lams_valid, T)
plt.ylabel("Transmittance T")
plt.xlabel("Wavelength (nm)")
plt.title("Transmittance")
plt.grid(alpha=0.3)

plt.subplot(2,2,3)
plt.plot(lams_valid, phi)
plt.ylabel("Phase φ (rad)")
plt.xlabel("Wavelength (nm)")
plt.title("Reflection Phase")
plt.grid(alpha=0.3)

plt.subplot(2,2,4)
plt.plot(lams_valid, np.abs(dphi_dlambda))
plt.ylabel(r"|dφ / dλ|  (rad / nm)")
plt.xlabel("Wavelength (nm)")
plt.title("Phase Slope (Group Delay)")
plt.yscale("log")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("zoomed_R_T_phase_dphi_nm.png", dpi=300)
plt.close()

print("✓ Saved: zoomed_R_T_phase_dphi_nm.png")
