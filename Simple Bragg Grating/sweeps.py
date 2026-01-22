import numpy as np
import matplotlib.pyplot as plt
import grcwa
from config import *


# ============================================================
# Solver for one wavelength
# ============================================================

def solver_system(nG, L1, L2, wavelength, theta, phi,
                  hs, hsio2, Nx, Ny, p, s):

    freq = 1.0 / wavelength   # grcwa uses freq = 1 / lambda
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)

    # Incident medium (air)
    obj.Add_LayerUniform(0.1, eair)

    # DBR stack
    for _ in range(5):
        obj.Add_LayerUniform(hs, es)
        obj.Add_LayerUniform(hsio2, esio2)

    # Exit medium (air)
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    # Plane-wave excitation (normal incidence)
    obj.MakeExcitationPlanewave(
        p_amp=p, p_phase=0,
        s_amp=s, s_phase=0,
        order=0
    )

    # Reflection / transmission (power only, sanity check)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

    # Modal amplitudes in incident medium
    Mi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    return Ri, Ti, Mi, obj


# ============================================================
# Reflection phase extraction (robust & general)
# ============================================================

def reflection_phase_from_amplitudes(Mi, tol=1e-9):
    """
    Extracts reflection phase from modal amplitudes.
    Mi = (ai, bi) returned by GetAmplitudes(layer=0)
    """
    ai, bi = Mi

    # Incident (0,0) eigenmode → only non-zero forward amplitude
    idx_inc = np.where(np.abs(ai) > tol)[0]
    if len(idx_inc) != 1:
        raise RuntimeError("Expected exactly one incident mode.")
    idx_inc = idx_inc[0]

    # Reflected (0,0) eigenmode → only non-zero backward amplitude
    idx_ref = np.where(np.abs(bi) > tol)[0]
    if len(idx_ref) != 1:
        raise RuntimeError("Expected exactly one reflected mode.")
    idx_ref = idx_ref[0]

    r = bi[idx_ref] / ai[idx_inc]
    return np.angle(r)


# ============================================================
# Wavelength sweep
# ============================================================

wavelengths = np.linspace(1.2, 2.1, 80)   # microns
phase = []
reflectance = []

for wl in wavelengths:
    Ri, Ti, Mi, obj = solver_system(
        nG=nG,
        L1=L1,
        L2=L2,
        wavelength=wl,
        theta=theta,
        phi=phi,
        hs=hs,
        hsio2=hsio2,
        Nx=Nx,
        Ny=Ny,
        p=1,     # p-polarized
        s=0
    )

    phi_r = reflection_phase_from_amplitudes(Mi)
    phase.append(phi_r)
    reflectance.append(Ri[0])   # zeroth-order reflectance

phase = np.unwrap(np.array(phase))
reflectance = np.array(reflectance)


# ============================================================
# Plot results
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(wavelengths, phase, lw=2)
plt.xlabel("Wavelength (μm)")
plt.ylabel("Reflection phase (rad)")
plt.title("Reflection phase vs wavelength (Si / SiO₂ DBR)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(wavelengths, reflectance, lw=2)
plt.xlabel("Wavelength (μm)")
plt.ylabel("Reflectance")
plt.title("Reflectance vs wavelength (Si / SiO₂ DBR)")
plt.grid(True)
plt.tight_layout()
plt.show()





