import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import time
import os
from datetime import datetime

BASE_DIR = r"D:\msc\Research\presentations\week14\Run6"
os.makedirs(BASE_DIR, exist_ok=True)





SAVE_DIR = r"D:\msc\Research\presentations\week14\Run6"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# GLOBAL SETTINGS
# =========================================================
N_PEAKS = 2
DBR_PAIRS = 5
TOP_K = 10

h_pattern = 0.3
hsio2_dbr = 0.2586
dlam = 1e-4

# =========================================================
# MATERIAL MODEL (Si)
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0], data.iloc[:,1],
            kind="cubic", bounds_error=False,
            fill_value="extrapolate"
        )
    n = _cache["li"](wavelength)
    return n**2

# =========================================================
# 3×3 METASURFACE
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    pattern = np.array(pattern).reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            f = np.clip(pattern[i,j],0,1)
            ep[(X>=i*dx)&(X<(i+1)*dx)&
               (Y>=j*dy)&(Y<(j+1)*dy)] = f*eps + (1-f)*eair
    return ep

# =========================================================
# RCWA SOLVER
# =========================================================
def solver_rt(lam, pattern, L_eff, hs_dbr, a):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG,[a,0],[0,a],1/lam,theta,phi,verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h_pattern, Nx, Ny)
        obj.Add_LayerUniform(L_eff, esio2)

        for _ in range(DBR_PAIRS):
            obj.Add_LayerUniform(hs_dbr, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)
        obj.MakeExcitationPlanewave(1,0,0,0)

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        
        
        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        
            
        r = bi[k0] / ai[k0]

        return complex(r)

    except Exception:
        return None
    



def solver_t(lam, pattern, L_eff, hs_dbr, a):
    try:
        
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG,[a,0],[0,a],1/lam,theta,phi,verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h_pattern, Nx, Ny)
        obj.Add_LayerUniform(L_eff, esio2)

        for _ in range(DBR_PAIRS):
            obj.Add_LayerUniform(hs_dbr, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)
        obj.MakeExcitationPlanewave(1,0,0,0)

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        

        Ri, Ti = obj.RT_Solve(normalize=1,byorder=1)
        

        return Ri[k0], Ti[k0]

    except Exception:
        return None, None


def solver_rte(lam, pattern, L_eff, hs_dbr, a):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG,[a,0],[0,a],1/lam,theta,phi,verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h_pattern, Nx, Ny)
        obj.Add_LayerUniform(L_eff, esio2)

        for _ in range(DBR_PAIRS):
            obj.Add_LayerUniform(hs_dbr, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep = get_epgrid_3x3(pattern, eps_si, a).flatten()
        obj.GridLayer_geteps(ep)
        obj.MakeExcitationPlanewave(1,0,0,0)

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        R,T = obj.RT_Solve(normalize=1,byorder=1)
        Sum = np.sum(R) + np.sum(T)
        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        
        r = bi[k0]/ai[k0]

        return complex(r), R[k0], T[k0],Sum

    except Exception:
        return None, None, None











# =========================================================
# PARAMETERS
# =========================================================
N_PARAMS = 12

bounds = np.array(
    [[0,1]]*9 +
    [[2, 4.8]] +     # L_eff
    [[0.20,0.28]] +    # hs_dbr
    [[0.5,1.5]]      # a
)

def decode(x):
    return bounds[:,0] + x*(bounds[:,1]-bounds[:,0])

# =========================================================
# TARGET WAVELENGTH WINDOWS
# =========================================================

# --- resonance 1 (1.52 µm) ---
flat1_L = np.linspace(1.500, 1.515, 7)
slope1  = np.linspace(1.515, 1.525, 9)
flat1_R = np.linspace(1.525, 1.540, 7)

flat2_L = np.linspace(1.565, 1.580, 7)
slope2  = np.linspace(1.580, 1.590, 9)
flat2_R = np.linspace(1.590, 1.605, 7)

lams_all = np.concatenate([
    slope1, slope2
])


# ---- index bookkeeping for wavelength regions ----
n1L = len(flat1_L)
n1S = len(slope1)
n1R = len(flat1_R)

n2L = len(flat2_L)
n2S = len(slope2)
n2R = len(flat2_R)

i = 0

idx_s1  = slice(i, i+n1S); i += n1S



idx_s2  = slice(i, i+n2S); i += n2S




lam_T1 = 1.5225   # center of first slope region
lam_T2 = 1.58   # center of second slope region


# =========================================================
# LOSS FUNCTION (TWO CLEAN RESONANCES)
# =========================================================




def loss_function(x, alpha=1.0, beta=1.0):

    eps = 1e-12

    # -------------------------------
    # decode geometry
    # -------------------------------
    p = decode(x)
    pattern = p[:9]
    L_eff   = p[9]
    hs_dbr  = p[10]
    a       = p[11]

    if not np.isfinite(L_eff) or L_eff < 3.0 or L_eff > 6.0:
        return 1e6

    # -------------------------------
    # spectral sweep
    # -------------------------------
    phis = []

    for lam in lams_all:
        r = solver_rt(lam, pattern, L_eff, hs_dbr, a)
        if r is None:
            return 1e6
        phis.append(np.angle(r))

    phis = np.unwrap(np.array(phis))

    # -------------------------------
    # slope helper (linear fit)
    # -------------------------------
    def get_slope(lams, phi):
        if len(phi) < 5:
            return 0.0
        A, _ = np.polyfit(lams, phi, 1)
        return abs(A)

    # -------------------------------
    # slopes in desired regions
    # -------------------------------
    S1 = get_slope(slope1, phis[idx_s1])
    S2 = get_slope(slope2, phis[idx_s2])

    # -------------------------------
    # LOSS
    # -------------------------------
    slope_reward   = -np.min([S1, S2])              # maximize both slopes
              # penalize mismatch

    
    penalty = 0
    if S1 < 1 or S2 < 1:
        penalty = 1e6

    loss = slope_reward + penalty   

    if not np.isfinite(loss):
        return 1e6

    print(
        f"S1={S1:.3e}, S2={S2:.3e}, "
        f"balance={(S1-S2)**2:.2e}, loss={loss:.3e}"
        
    )

    return loss







def save_elite_geometries(elite_solutions, tag="run"):
    """
    Saves elite geometries to:
    - one .npy file
    - individual .txt files
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- Save NPY ----------
    npy_path = os.path.join(
        SAVE_DIR,
        f"elite_geometries_{tag}_{timestamp}.npy"
    )

    np.save(npy_path, np.array(elite_solutions, dtype=object))

    # ---------- Save TXT (one per geometry) ----------
    for i, (loss, params) in enumerate(elite_solutions):

        txt_path = os.path.join(
            SAVE_DIR,
            f"geom_{i}_loss_{loss:.3e}_{timestamp}.txt"
        )

        pattern = params[:9]
        L_eff   = params[9]
        hs_dbr  = params[10]
        a       = params[11]

        with open(txt_path, "w") as f:
            f.write(f"Geometry {i}\n")
            f.write(f"Loss = {loss:.6e}\n\n")

            f.write("Pattern (9 params):\n")
            for j, p in enumerate(pattern):
                f.write(f"  p{j+1} = {p:.6f}\n")

            f.write("\nOther parameters:\n")
            f.write(f"  L_eff  = {L_eff:.6f}\n")
            f.write(f"  hs_dbr = {hs_dbr:.6f}\n")
            f.write(f"  a      = {a:.6f}\n")

    print(f"Saved {len(elite_solutions)} geometries:")
    print(f"  → {npy_path}")
    print(f"  → individual .txt files in {SAVE_DIR}")








# =========================================================
# CMA-ES WITH TOP-10 TRACKING
# =========================================================
elite_solutions = []

x0 = np.full(N_PARAMS, 0.5)
es = cma.CMAEvolutionStrategy(
    x0, 0.3,
    {"popsize": 10, "maxiter": 40}
)

while not es.stop():
    xs = es.ask()
    losses = []

    for x in xs:
        loss = loss_function(x)
        losses.append(loss)

        if np.isfinite(loss):
            elite_solutions.append((loss, decode(x).copy()))
            elite_solutions.sort(key=lambda t: t[0])
            elite_solutions[:] = elite_solutions[:TOP_K]

    es.tell(xs, losses)

# ---------- SAVE RESULTS ----------
save_elite_geometries(elite_solutions, tag="Run6")



def find_resonance_in_range(lams, T, lam_min, lam_max):
    lams = np.array(lams)
    T = np.array(T)

    if T.size == 0:
        return None, None

    mask = (lams >= lam_min) & (lams <= lam_max)
    if not np.any(mask):
        return None, None

    idx = np.argmax(T[mask])
    return lams[mask][idx], T[mask][idx]




def build_solver(lam, pattern, L_eff, hs_dbr, a):
    eps_si = epsilon_lambda(lam)
    obj = grcwa.obj(nG, [a, 0], [0, a], 1/lam, theta, phi, verbose=0)

    # -------------------------
    # Layer stack
    # -------------------------
    obj.Add_LayerUniform(0.1, eair)

    obj.Add_LayerGrid(h_pattern, Nx, Ny)   # grid layer 1
    obj.Add_LayerGrid(L_eff, Nx, Ny)       # grid layer 2 (cavity)

    for _ in range(DBR_PAIRS):
        obj.Add_LayerUniform(hs_dbr, eps_si)
        obj.Add_LayerUniform(hsio2_dbr, esio2)

    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    # -------------------------
    # ε grids (IMPORTANT PART)
    # -------------------------

    ep_pattern = get_epgrid_3x3(pattern, eps_si, a).flatten()
    ep_cavity  = np.full(Nx * Ny, esio2)

    # concatenate in layer order
    ep_all = np.concatenate([ep_pattern, ep_cavity])

    obj.GridLayer_geteps(ep_all)

    # -------------------------
    # Excitation
    # -------------------------
    obj.MakeExcitationPlanewave(1, 0, 0, 0)

    return obj





def compute_xz_field_intensity(obj, which_layer, z_min, z_max, Nz, y_index=None):
    """
    Returns:
    x (Nx)
    z (Nz)
    I (Nz, Nx)
    """

    z_vals = np.linspace(z_min, z_max, Nz)
    I_xz = np.zeros((Nz, Nx))

    if y_index is None:
        y_index = Ny // 2

    for i, z in enumerate(z_vals):
        E, _ = obj.Solve_FieldOnGrid(which_layer=which_layer, z_offset=z)

        Ex, Ey, Ez = E
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        I_xz[i, :] = I[:, y_index]

    x = np.linspace(0, a, Nx)
    return x, z_vals, I_xz





# =========================================================
# PLOTTING
# =========================================================
def plot_geometry_response(params, idx):
    pattern = params[:9]
    L_eff   = params[9]
    hs_dbr  = params[10]
    a       = params[11]

    lams = np.linspace(1.48, 1.62, 600)
    R, T, phis, Sum = [], [], [], []

    for lam in lams:
        out = solver_rte(lam, pattern, L_eff, hs_dbr, a)
        if out is None or out[0] is None:
            R.append(0.0)
            T.append(0.0)
            phis.append(0.0)
            Sum.append(0.0)
            continue
        r, R0, T0, Sum0 = out
        R.append(R0)
        T.append(T0)
        phis.append(np.angle(r))
        Sum.append(Sum0)

    phis = np.unwrap(phis)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # --- Spectrum plot ---
    plt.figure()
    plt.plot(lams, T, label="T")
    plt.plot(lams, R, label="R")
    plt.plot(lams, Sum, label="Sum (R+T)", linestyle="--", alpha=0.7)
    plt.legend()
    plt.xlabel("λ (µm)")
    plt.ylabel("Response")
    plt.title(f"Geometry {idx}")
    plt.grid(alpha=0.3)

    fname = os.path.join(BASE_DIR, f"geom_{idx}_spectrum_{timestamp}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Phase plot ---
    plt.figure()
    plt.plot(lams, phis)
    plt.xlabel("λ (µm)")
    plt.ylabel("arg(r)")
    plt.title(f"Phase — Geometry {idx}")
    plt.grid(alpha=0.3)

    fname = os.path.join(BASE_DIR, f"geom_{idx}_phase_{timestamp}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    return lams, T

    


def plot_xz_intensity(x, z, I, title, fname):
    plt.figure(figsize=(6,4))
    plt.pcolormesh(x, z, I, shading='auto', cmap='inferno')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar(label="|E|²")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()





# =========================================================
# GENERATE PLOTS FOR TOP-10 + FIELD MAPS
# =========================================================

for i, (loss, params) in enumerate(elite_solutions):

    print(f"\n=== Geometry {i} | loss={loss:.3e} ===")

    pattern = params[:9]
    L_eff   = params[9]
    hs_dbr  = params[10]
    a       = params[11]

    # --- Transmission spectrum ---
    lams, T = plot_geometry_response(params, i)
    if np.all(np.array(T) == 0):
        print(f"Geometry {i}: no valid transmission, skipping field plots.")
        continue

    # --- Find resonances ---
    lam1, T1 = find_resonance_in_range(lams, T, 1.510, 1.530)
    lam2, T2 = find_resonance_in_range(lams, T, 1.570, 1.590)

    resonances = [(lam1, T1), (lam2, T2)]

    for j, (lam, Tval) in enumerate(resonances):

        if lam is None:
            continue

        print(f"  Resonance {j+1}: λ = {lam:.5f}, T = {Tval:.3f}")

        obj = build_solver(
            lam=lam,
            pattern=pattern,
            L_eff=L_eff,
            hs_dbr=hs_dbr,
            a=a
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # --- Patterned layer ---
        x, z, Ixz = compute_xz_field_intensity(
            obj,
            which_layer=1,
            z_min=0,
            z_max=h_pattern,
            Nz=80
        )

        fname = os.path.join(
            BASE_DIR,
            f"geom_{i}_res_{j+1}_pattern_{timestamp}.png"
        )

        plot_xz_intensity(
            x, z, Ixz,
            title=f"Geom {i} | Pattern | λ={lam:.4f}",
            fname=fname
        )

        # --- Cavity ---
        x, z, Ixz = compute_xz_field_intensity(
            obj,
            which_layer=2,
            z_min=0,
            z_max=L_eff,
            Nz=120
        )

        fname = os.path.join(
            BASE_DIR,
            f"geom_{i}_res_{j+1}_cavity_{timestamp}.png"
        )

        plot_xz_intensity(
            x, z, Ixz,
            title=f"Geom {i} | Cavity | λ={lam:.4f}",
            fname=fname
        )

