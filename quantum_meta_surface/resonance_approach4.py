import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# STORE TOP CANDIDATES
# =========================================================
TOP_K = 10
top_candidates = []

def update_top_candidates(loss, params):
    global top_candidates
    top_candidates.append((loss, params.copy()))
    top_candidates.sort(key=lambda x: x[0])
    top_candidates = top_candidates[:TOP_K]

# =========================================================
# TARGET POLES (µm)
# =========================================================
lambda0_A1 = 1.5
lambda0_A2 = 0.75

# =========================================================
# WAVELENGTH GRIDS
# =========================================================
lambdas_A1 = np.concatenate([
    np.linspace(1.485, 1.495, 6),
    np.linspace(1.495, 1.505, 16),
    np.linspace(1.505, 1.515, 6)
])

lambdas_A2 = np.concatenate([
    np.linspace(0.735, 0.745, 6),
    np.linspace(0.745, 0.755, 16),
    np.linspace(0.755, 0.765, 6)

])



history = {
    "loss": [],
    "T1": [],
    "T2": [],
    "dphi1": [],
    "dphi2": []
}

# =========================================================
# MATERIAL MODEL ε(λ)  (LOSSLESS, CMA-SAFE)
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
        n = 3.6     # real silicon in visible
    else:
        n = _cache["li"](wavelength)

    return n**2

# =========================================================
# 3×3 ε GRID
# =========================================================
def get_epgrid_3x3(pattern, et, a):
    pattern = np.clip(pattern.reshape(3,3), 0.05, 0.95)
    epgrid = np.ones((Nx, Ny), dtype=complex) * eair

    dx = a / 3
    dy = a / 3
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            mask = (
                (X >= i*dx) & (X < (i+1)*dx) &
                (Y >= j*dy) & (Y < (j+1)*dy)
            )
            epgrid[mask] = eair + pattern[i,j] * (et - eair)

    return epgrid

# =========================================================
# RCWA SOLVER (RETURNS r, R, T)
# =========================================================
def solver_system(f, pattern, hpat, hsio2, hs, a):

    try:
        L1 = [a, 0]
        L2 = [0, a]
        eps_si = epsilon_lambda(1/f)

        obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(hpat, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        epgrid = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(epgrid)

        obj.MakeExcitationPlanewave(
            p_amp=1, p_phase=0,
            s_amp=0, s_phase=0,
            order=0
        )


        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        # --- stable reflection coefficient ---
        if abs(ai[k0]) > abs(ai[k0+nV]):
            den = ai[k0]
            num = bi[k0]
        else:
            den = ai[k0+nV]
            num = bi[k0+nV]

        if abs(den) < 1e-12:
            return None

        r = num / den
        if not np.isfinite(r):
            return None

        # --- power coefficients ---
        R, T = obj.RT_Solve(normalize=1, byorder=1)
        return r, R[0], T[0]

    except Exception:
        return None

# =========================================================
# PHASE
# =========================================================
def compute_phase(params, lambdas):
    phis = []
    for lam in lambdas:
        out = solver_system(1/lam, params[:9], params[9], params[10], params[11], params[12])
        if out is None:
            return None
        r, _, _ = out
        phis.append(np.angle(r))
    return np.unwrap(np.array(phis))

# =========================================================
# TRANSMITTANCE PENALTY
# =========================================================
def trans_penalty(T, Tmin=0.8):
    return max(0.0, Tmin - T)**2

# =========================================================
# LOSS FUNCTION (PHASE + Q + TRANSMITTANCE)
# =========================================================
def loss_double_sqrt_pole(x):

    params = decode(x)

    phi1 = compute_phase(params, lambdas_A1)
    phi2 = compute_phase(params, lambdas_A2)
    if phi1 is None or phi2 is None:
        return 1e6

    phi1 -= phi1[np.argmin(np.abs(lambdas_A1 - lambda0_A1))]
    phi2 -= phi2[np.argmin(np.abs(lambdas_A2 - lambda0_A2))]

    mask1 = lambdas_A1 > lambda0_A1 + 0.003
    mask2 = lambdas_A2 > lambda0_A2 + 0.003
    if np.sum(mask1) < 4 or np.sum(mask2) < 4:
        return 1e6

    tgt1 = np.sqrt(lambdas_A1[mask1] - lambda0_A1)
    tgt2 = np.sqrt(lambdas_A2[mask2] - lambda0_A2)

    A1 = np.dot(phi1[mask1], tgt1) / np.dot(tgt1, tgt1)
    A2 = np.dot(phi2[mask2], tgt2) / np.dot(tgt2, tgt2)

    fit1 = np.mean((phi1[mask1] - A1 * tgt1)**2)
    fit2 = np.mean((phi2[mask2] - A2 * tgt2)**2)

    dphi1 = np.gradient(phi1, lambdas_A1, edge_order=1)
    dphi2 = np.gradient(phi2, lambdas_A2, edge_order=1)

    p1 = -np.nanmax(np.abs(dphi1))
    p2 = -np.nanmax(np.abs(dphi2))

    # --- Transmission at λ0s ---
    Tvals = []
    for lam in (lambda0_A1, lambda0_A2):
        out = solver_system(
            1/lam,
            params[:9], params[9], params[10], params[11], params[12]
        )
        if out is None:
            return 1e6
        _, _, T = out
        Tvals.append(T)

    T_target = 0.8
    Tpen = sum(max(0.0, T_target - T)**2 for T in Tvals)

    loss = (fit1 + fit2) + 3*(p1 + p2) + 5*Tpen

    global last_eval_info
    last_eval_info = {
        "loss": loss,
        "fit": fit1 + fit2,
        "dphi1": abs(p1),
        "dphi2": abs(p2),
        "T1": Tvals[0],
        "T2": Tvals[1],
    }

    update_top_candidates(loss, params)
    return loss



# =========================================================
# BOUNDS
# =========================================================
bounds = np.array(
    [[0,1]]*9 +
    [[0.05,0.40],
     [0.05,0.60],
     [0.22,0.28],
     [0.1,0.7]]
)

def decode(x):
    return bounds[:,0] + np.clip(x,0,1)*(bounds[:,1]-bounds[:,0])

# =========================================================
# CMA SETUP
# =========================================================
x0_phys = np.array([0.5]*9 + [0.15, 0.35, 0.25, 0.3])
x0 = (x0_phys - bounds[:,0])/(bounds[:,1]-bounds[:,0])

es = cma.CMAEvolutionStrategy(
    x0, 0.25,
    {"popsize":10, "maxiter":60, "verb_disp":1}
)

# =========================================================
# RUN CMA
# =========================================================
while not es.stop():
    xs = es.ask()
    losses = [loss_double_sqrt_pole(x) for x in xs]
    es.tell(xs, losses)

    info = last_eval_info
    print(
        f"ITER {es.countiter:03d} | "
        f"LOSS={info['loss']:.2e} | "
        f"T(1.5)={info['T1']:.3f}, T(0.75)={info['T2']:.3f} | "
        f"|dφ|1={info['dphi1']:.2e}, |dφ|2={info['dphi2']:.2e}"
    )

# =========================================================
# FINAL REPORT (INCLUDES R,T @ 0.75 µm)
# =========================================================
# =========================================================
# FINAL REPORT: TOP 10 GEOMETRIES
# =========================================================
print("\n=========== TOP 10 RESULTS ===========")

for i, (loss, params) in enumerate(top_candidates):

    print(f"\n================= GEOMETRY {i+1} =================")
    print(f"LOSS = {loss:.3e}")
    print("Pattern (3×3):")
    print(params[:9].reshape(3,3))
    print(
        f"hpat = {params[9]:.3f} µm | "
        f"hsio2 = {params[10]:.3f} µm | "
        f"hs = {params[11]:.3f} µm | "
        f"a = {params[12]:.3f} µm"
    )

    # -------------------------------------------------
    # R, T at λ = 0.75 µm
    # -------------------------------------------------
    out75 = solver_system(
        1/lambda0_A2,
        params[:9],
        params[9],
        params[10],
        params[11],
        params[12]
    )

    if out75 is not None:
        _, R75, T75 = out75
        print(f"@ 0.75 µm → R = {R75:.3f}, T = {T75:.3f}")
    else:
        print("@ 0.75 µm → solver failed")

    # -------------------------------------------------
    # PHASE PLOTS
    # -------------------------------------------------
    phi1 = compute_phase(params, lambdas_A1)
    phi2 = compute_phase(params, lambdas_A2)

    # -------------------------------------------------
    # R / T SPECTRA (BOTH RANGES)
    # -------------------------------------------------
    lambdas_RT_1 = np.linspace(1.45, 1.55, 120)
    lambdas_RT_2 = np.linspace(0.70, 0.80, 120)

    R1, T1 = [], []
    R2, T2 = [], []

    for lam in lambdas_RT_1:
        out = solver_system(
            1/lam,
            params[:9],
            params[9],
            params[10],
            params[11],
            params[12]
        )
        if out is None:
            R1.append(np.nan)
            T1.append(np.nan)
        else:
            _, Ri, Ti = out
            R1.append(Ri)
            T1.append(Ti)

    for lam in lambdas_RT_2:
        out = solver_system(
            1/lam,
            params[:9],
            params[9],
            params[10],
            params[11],
            params[12]
        )
        if out is None:
            R2.append(np.nan)
            T2.append(np.nan)
        else:
            _, Ri, Ti = out
            R2.append(Ri)
            T2.append(Ti)

    R1, T1 = np.array(R1), np.array(T1)
    R2, T2 = np.array(R2), np.array(T2)

    # -------------------------------------------------
    # PLOTTING
    # -------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18,4))

    # ---- Phase ----
    axs[0].plot(lambdas_A1, phi1, 'o-', label="Phase @ 1.5 µm band")
    axs[0].plot(lambdas_A2, phi2, 's-', label="Phase @ 0.75 µm band")
    axs[0].axvline(lambda0_A1, ls=":", c="r")
    axs[0].axvline(lambda0_A2, ls=":", c="r")
    axs[0].set_xlabel("Wavelength (µm)")
    axs[0].set_ylabel("Phase (rad)")
    axs[0].set_title("Reflection Phase")
    axs[0].legend()
    axs[0].grid(True)

    # ---- R/T near 1.5 µm ----
    axs[1].plot(lambdas_RT_1, R1, label="R")
    axs[1].plot(lambdas_RT_1, T1, label="T")
    axs[1].axvline(lambda0_A1, ls=":", c="r")
    axs[1].set_xlabel("Wavelength (µm)")
    axs[1].set_ylabel("Power")
    axs[1].set_title("R / T near 1.5 µm")
    axs[1].legend()
    axs[1].grid(True)

    # ---- R/T near 0.75 µm ----
    axs[2].plot(lambdas_RT_2, R2, label="R")
    axs[2].plot(lambdas_RT_2, T2, label="T")
    axs[2].axvline(lambda0_A2, ls=":", c="r")
    axs[2].set_xlabel("Wavelength (µm)")
    axs[2].set_ylabel("Power")
    axs[2].set_title("R / T near 0.75 µm")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

