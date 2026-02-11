import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import grcwa
from config import *

# =========================================================
# WAVELENGTH SWEEP RANGES (µm)
# =========================================================
lambdas_vis = np.linspace(0.70, 0.80, 120)
lambdas_ir  = np.linspace(1.49980, 1.50020, 120)

# =========================================================
# FIXED DBR DESIGN (DESIGNED AT 1.5 µm)
# =========================================================
n_sio2 = 1.45
lambda_dbr = 1.50
#hsio2_dbr = lambda_dbr / (4 * n_sio2)   # ≈ 0.259 µm

print(f"Using fixed hsio2_dbr = {hsio2_dbr:.4f} µm")

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
# 3×3 ε GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = (np.array(pattern).reshape(3,3) > 0.5).astype(int)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                mask = (
                    (X >= i*dx) & (X < (i+1)*dx) &
                    (Y >= j*dy) & (Y < (j+1)*dy)
                )
                ep[mask] = eps

    return ep


# =========================================================
# RCWA SOLVER (SAFE + CORRECT PHASE)
# =========================================================
def solver_system(f, p1, p2, h1, h2, hs, hsio2_spacer, a):
    try:
        eps_si = epsilon_lambda(1/f)
        obj = grcwa.obj(nG, [a,0], [0,a], f, theta, phi, verbose=0)

        # Top air
        obj.Add_LayerUniform(0.1, eair)

        # Patterned layer 1
        obj.Add_LayerGrid(h1, Nx, Ny)

        # SiO2 spacer (OPTIMIZED / FIXED INPUT)
        obj.Add_LayerUniform(hsio2_spacer, esio2)

        # Patterned layer 2
        obj.Add_LayerGrid(h2, Nx, Ny)

        # DBR STACK (FIXED hsio2_dbr)
        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        # Bottom air
        obj.Add_LayerUniform(0.1, eair)

        obj.Init_Setup()

        ep1 = get_epgrid_3x3(p1, eps_si, a).flatten()
        ep2 = get_epgrid_3x3(p2, eps_si, a).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(
            p_amp=1, p_phase=0,
            s_amp=0, s_phase=0
        )

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

        k0 = np.where((obj.G[:,0]==0) & (obj.G[:,1]==0))[0][0]
        nV = obj.nG

        if abs(ai[k0]) > abs(ai[k0+nV]):
            r00 = bi[k0] / ai[k0]
        else:
            r00 = bi[k0+nV] / ai[k0+nV]

        return r00, Ri[k0], Ti[k0]

    except Exception:
        # Singular RCWA point → skip safely
        return np.nan, np.nan, np.nan


# =========================================================
# SWEEP FUNCTION (ROBUST)
# =========================================================
def sweep_geometry(p1, p2, h1, h2, hs, hsio2_spacer, a, lambdas):

    phases = np.full(len(lambdas), np.nan)
    Rs     = np.full(len(lambdas), np.nan)
    Ts     = np.full(len(lambdas), np.nan)

    for i, lam in enumerate(lambdas):
        r00, R00, T00 = solver_system(
            1/lam, p1, p2, h1, h2, hs, hsio2_spacer, a
        )

        if np.isfinite(r00):
            phases[i] = np.angle(r00)
            Rs[i] = R00
            Ts[i] = T00

    # unwrap only valid phase points
    mask = np.isfinite(phases)
    phases[mask] = np.unwrap(phases[mask])

    return phases, Rs, Ts


# =========================================================
# DEFINE FIXED GEOMETRY (PASTE YOUR OPTIMIZED ONE HERE)
# =========================================================
p1 = [0,0,0,
      1,1,1,
      0,0,0]

p2 = [1,1,0,
      0,1,1,
      0,0,1]

h1 = 0.19014648
h2 = 0.07730069
hs = 0.29039493   # Si thickness in DBR
hsio2_spacer = 0.14386821  # SiO2 between patterned layers
a = 1.4641

print("\n=== SWEEPING FIXED GEOMETRY ===")
print(f"h1={h1}, h2={h2}, hs={hs}, hsio2_spacer={hsio2_spacer}, a={a}")


# =========================================================
# RUN SWEEPS
# =========================================================
phi_vis, R_vis, T_vis = sweep_geometry(
    p1, p2, h1, h2, hs, hsio2_spacer, a, lambdas_vis
)

phi_ir, R_ir, T_ir = sweep_geometry(
    p1, p2, h1, h2, hs, hsio2_spacer, a, lambdas_ir
)


# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(7,5))
#plt.plot(lambdas_vis, phi_vis, label="Phase (0.7–0.8 µm)")
plt.plot(lambdas_ir,  phi_ir,  label="Phase (1.45–1.55 µm)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection Phase (rad)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
#plt.plot(lambdas_vis, R_vis, label="R (0.7–0.8 µm)")
plt.plot(lambdas_ir,  R_ir,  label="R (1.45–1.55 µm)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
#plt.plot(lambdas_vis, T_vis, label="T (0.7–0.8 µm)")
plt.plot(lambdas_ir,  T_ir,  label="T (1.45–1.55 µm)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmittance")
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# OPTIONAL DIAGNOSTICS
# =========================================================
A_vis, _ = np.polyfit(lambdas_vis[np.isfinite(phi_vis)],
                      phi_vis[np.isfinite(phi_vis)], 1)
A_ir,  _ = np.polyfit(lambdas_ir[np.isfinite(phi_ir)],
                      phi_ir[np.isfinite(phi_ir)], 1)

print("\n=== DISPERSION METRICS ===")
print(f"Phase slope (vis) = {A_vis:.3e} rad/µm")
print(f"Phase slope (IR)  = {A_ir:.3e} rad/µm")

plt.figure(figsize=(7,5))
plt.plot(lambdas_vis, R_vis + T_vis, label="R+T (vis)")
plt.plot(lambdas_ir,  R_ir  + T_ir,  label="R+T (IR)")
plt.axhline(1.0, ls="--", c="k")
plt.xlabel("Wavelength (µm)")
plt.ylabel("R + T")
plt.legend()
plt.grid(True)
plt.show()
