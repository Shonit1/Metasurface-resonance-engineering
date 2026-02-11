import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import grcwa
from config import *

# =========================================================
# BIC GEOMETRY (from optimizer)
# =========================================================
h1_0   = 0.284860
h2_0   = 0.083065
hs     = 0.298298
hsio2 = 0.278357

P1 = np.array([[0,0,0],[1,1,1],[0,0,0]])
P2 = np.array([[1,1,0],[0,1,1],[0,0,1]])

a_fixed = 1.4641
lambda0 = 1.5  # µm

# =========================================================
# SYMMETRY BREAKING
# =========================================================
epsilons = np.array([0.0, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
lams = np.linspace(lambda0 - 2e-3, lambda0 + 2e-3, 2001)

Q_list = []
eps2_list = []

# =========================================================
# MATERIAL MODEL
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
# GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    ep = np.ones((Nx,Ny), dtype=complex) * eair
    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[(X>=i*dx)&(X<(i+1)*dx)&
                   (Y>=j*dy)&(Y<(j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVERS
# =========================================================
def solver_r00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG, [a_fixed,0], [0,a_fixed],
                        1/lam, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
        ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        return bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
               else bi[k0+nV]/ai[k0+nV]
    except:
        return None

def solver_t00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG, [a_fixed,0], [0,a_fixed],
                        1/lam, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
        ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)

        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        return Ti[k0]
    except:
        return None

# =========================================================
# LORENTZIAN
# =========================================================
def lorentzian(lam, lam0, gamma, A, C):
    return A / ((lam - lam0)**2 + gamma**2) + C

# =========================================================
# MAIN LOOP
# =========================================================
for eps in epsilons:

    print(f"\n=== ε = {eps:.1e} ===")
    h1 = h1_0 + eps
    h2 = h2_0 - eps

    Rs, Ts, phis, lams_valid = [], [], [], []

    for lam in lams:
        r = solver_r00(lam, h1, h2, hs, hsio2)
        t = solver_t00(lam, h1, h2, hs, hsio2)
        if r is None or t is None:
            continue

        Rs.append(abs(r)**2)
        Ts.append(t)
        phis.append(np.angle(r))
        lams_valid.append(lam)

    Rs = np.array(Rs)
    Ts = np.array(Ts)
    phis = np.unwrap(np.array(phis))
    lams_valid = np.array(lams_valid)

    peaks, _ = find_peaks(Rs, prominence=0.01)
    if len(peaks) == 0:
        print("  ⚠ No resonance")
        continue

    peak = peaks[np.argmax(Rs[peaks])]
    lam_peak = lams_valid[peak]

    mask = np.abs(lams_valid - lam_peak) < 4e-4
    popt, _ = curve_fit(
        lorentzian,
        lams_valid[mask],
        Rs[mask],
        p0=[lam_peak, 1e-4, 1.0, np.min(Rs)]
    )

    lam0_fit, gamma, _, _ = popt
    Q = lam0_fit / (2*gamma)

    print(f"  λ₀={lam0_fit:.6f}, γ={gamma:.2e}, Q={Q:.2e}")

    if eps > 0:
        Q_list.append(Q)
        eps2_list.append(eps**2)

    # =====================================================
    # 3-PANEL PLOT
    # =====================================================
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.plot(lams_valid, Rs)
    plt.title("Reflectance")
    plt.xlabel("λ (µm)")
    plt.ylabel("R")

    plt.subplot(1,3,2)
    plt.plot(lams_valid, Ts)
    plt.title("Transmittance")
    plt.xlabel("λ (µm)")
    plt.ylabel("T")

    plt.subplot(1,3,3)
    plt.plot(lams_valid, phis)
    plt.title("Phase (reflection)")
    plt.xlabel("λ (µm)")
    plt.ylabel("φ")

    plt.suptitle(f"ε={eps:.1e}, Q≈{Q:.1e}")
    plt.tight_layout()
    plt.savefig(f"resonance_eps_{eps:.1e}.png", dpi=200)
    plt.close()

    print(f"  ✓ Saved resonance_eps_{eps:.1e}.png")

# =========================================================
# Q vs ε²
# =========================================================
plt.figure(figsize=(6,4))
plt.loglog(eps2_list, Q_list, "o-", label="Data")
plt.loglog(eps2_list, Q_list[0]*eps2_list[0]/np.array(eps2_list),
           "--", label="~1/ε²")
plt.xlabel("ε²")
plt.ylabel("Q")
plt.legend()
plt.tight_layout()
plt.savefig("Q_vs_eps2.png", dpi=200)
plt.close()

print("\n✓ Saved Q_vs_eps2.png")
