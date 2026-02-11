import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import grcwa
from config import *

'''For BIC resonance reduce it dlam_um  3e-4 and take hints from bic_symmetry.py plots.  '''



# =========================================================
# USER INPUT: GEOMETRY (BEST FOUND)
# =========================================================
h1    = 0.275283
h2    = 0.085651
hs    = 0.298414
hsio2 = 0.284498

# =========================================================
# LATTICE & PATTERNS
# =========================================================
P1 = np.array([
    [0,0,0],
    [1,1,1],
    [0,0,0]
])

P2 = np.array([
    [1,1,0],
    [0,1,1],
    [0,0,1]
])

a_fixed = 1.4641

# =========================================================
# RESONANCE CENTER & ZOOM (µm → nm)
# =========================================================
lambda0_um = 1.50090         # center wavelength (µm)
dlam_um    = 3e-5            # ±30 pm window
Npts       = 1001             # ultra-dense grid

lams_um = np.linspace(
    lambda0_um - dlam_um,
    lambda0_um + dlam_um,
    Npts
)

lams_nm = lams_um * 1000.0    # convert to nm

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
# 3×3 PERMITTIVITY GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    dx = dy = a / 3

    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i, j]:
                ep[(X >= i*dx) & (X < (i+1)*dx) &
                   (Y >= j*dy) & (Y < (j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVERS
# =========================================================
def solver_rt(lam):
    try:
        eps_si = epsilon_lambda(lam)

        obj = grcwa.obj(
            nG, [a_fixed,0], [0,a_fixed],
            1/lam, theta, phi, verbose=0
        )

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)

        # 🔑 CRITICAL LINE
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
        ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
        

        if obj.G.size == 0:
            return None, None

        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        r = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) else bi[k0+nV]/ai[k0+nV]
        t = Ti[k0]

        return r, Ri,Ti,obj

    except Exception as e:
        return None, None
    


def build_solver(lam, p1,p2, a):
    eps_si = epsilon_lambda(lam)
    obj = grcwa.obj(nG, [a, 0], [0, a], 1/lam, theta, phi, verbose=0)

    # -------------------------
    # Layer stack
    # -------------------------
    obj.Add_LayerUniform(0.1, eair)

    obj.Add_LayerGrid(h1, Nx, Ny)   # grid layer 1
    obj.Add_LayerGrid(hsio2, Nx, Ny) # grid layer 2 (spacer)
    obj.Add_LayerGrid(h2, Nx, Ny)       # grid layer 3 (cavity)






    for _ in range(5):
        obj.Add_LayerGrid(hs, Nx, Ny)          # patterned Si layers
        obj.Add_LayerGrid(hsio2_dbr, Nx, Ny)

    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    # -------------------------
    # ε grids (IMPORTANT PART)
    # -------------------------

    ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
    ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
    epSio2_cavity  = np.full(Nx * Ny, esio2)
    epSi_cavity = np.full(Nx * Ny, eps_si)

    # concatenate in layer order
    ep_all = np.concatenate([ep1, epSio2_cavity, ep2] + [epSi_cavity, epSio2_cavity]*5)

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




def find_resonance_in_range(lams, r_list):
    lams = np.array(lams)
    r_list = np.array(r_list)

    

    idx = np.argmin(r_list)
    return lams[idx], r_list[idx]




Ri_list, Ti_list,T, phi_list, lams_valid,sum,r_list = [], [], [], [],[],[],[]

for lam_um, lam_nm in zip(lams_um, lams_nm):
    r,Ri,Ti,obj = solver_rt(lam_um)
    
    Ri_list.append(Ri)
    Ti_list.append(Ti)
    T.append(np.real(Ti[0]))
    phi_list.append(np.angle(r))
    lams_valid.append(lam_nm)
    sum.append(np.sum(Ri) + np.sum(Ti))
    r_list.append(np.abs(r)**2)


R00 = np.array(Ri_list)[:,0]
t00 = np.array(Ti_list)[:,0]

#R10 = np.array(Ri_list)[:,1]
#T10 = np.array(Ti_list)[:,1]
#R01 = np.array(Ri_list)[:,2]
#T01 = np.array(Ti_list)[:,2]
plt.figure(figsize=(14,8))
plt.plot(lams_valid, R00, label="R00")
plt.plot(lams_valid, t00, label="T00")
plt.plot(lams_valid, r_list, label="(abs(r))**2,r = bi[k0]/ai[k0]", color="red", linestyle="--")
#plt.plot(lams_valid, R10, label="R10")
#plt.plot(lams_valid, T10, label="T10")
#plt.plot(lams_valid, R01, label="R01")
#plt.plot(lams_valid, T01, label="T01")
plt.plot(lams_valid, sum, label="Sum of R and T", color="black", linestyle="--")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance/Transmittance")
plt.title("Diffraction Orders vs Wavelength")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
    
    



def plot_xz_intensity(x, z, I, title):
    plt.figure(figsize=(6,4))
    plt.pcolormesh(x, z, I, shading='auto', cmap='inferno')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar(label="|E|²")
    plt.title(title)
    plt.tight_layout()
    plt.show()





# =========================================================
# GENERATE PLOTS FOR TOP-10 + FIELD MAPS
# =========================================================




# --- Find resonances ---
lam1, r_list = find_resonance_in_range(lams_um , r_list)
    

resonances = [(lam1, r_list)]



obj = build_solver(
            lam=lam1,
            p1=P1,
            p2=P2,
            a=a_fixed
        )

'''

for i,h in enumerate([h1, hsio2, h2]+[hs, hsio2_dbr]*5):

        # --- Patterned layer ---
    x, z, Ixz = compute_xz_field_intensity(
            obj,
            which_layer=i+1,
            z_min=0,
            z_max=h,
            Nz=80
        )

        

    plot_xz_intensity(
            x, z, Ixz,
            title=f"Geom {i} | Pattern | λ={lam1:.4f}",
            
        )

     '''  






layer_thicknesses = (
    [h1, hsio2, h2] +
    [hs, hsio2_dbr] * 5
)

layer_names = (
    ["Si pattern", "SiO2 spacer", "Cavity"] +
    ["Si DBR", "SiO2 DBR"] * 5
)





def compute_full_xz_intensity(obj, layer_thicknesses, Nz_per_layer=80):
    """
    Returns:
    x (Nx)
    z (Nz_total)
    I (Nz_total, Nx)
    layer_bounds (list of z positions)
    """

    I_all = []
    z_all = []
    z_offset_global = 0.0
    layer_bounds = []

    for i, h in enumerate(layer_thicknesses):
        x, z_local, I_local = compute_xz_field_intensity(
            obj,
            which_layer=i+1,
            z_min=0,
            z_max=h,
            Nz=Nz_per_layer
        )

        z_shifted = z_local + z_offset_global

        I_all.append(I_local)
        z_all.append(z_shifted)

        z_offset_global += h
        layer_bounds.append(z_offset_global)

    I_all = np.vstack(I_all)
    z_all = np.concatenate(z_all)

    return x, z_all, I_all, layer_bounds




def plot_full_xz_intensity(x, z, I, layer_bounds, title, fname):
    plt.figure(figsize=(6, 6))

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




x, z, Ixz, layer_bounds = compute_full_xz_intensity(
    obj,
    layer_thicknesses,
    Nz_per_layer=80
)

plot_full_xz_intensity(
    x, z, Ixz,
    layer_bounds,
    title=f"Full structure | λ={lam1:.6f} µm",
    fname="full_xz_field.png"
)


















'''


# =========================================================
# RCWA SCAN
# =========================================================
R_list, T_list, phi_list, lams_valid = [], [], [], []

for lam_um, lam_nm in zip(lams_um, lams_nm):
    r,t = solver_rt(lam_um)
   

    if r is None or t is None:
        continue

    R_list.append(abs(r)**2)
    T_list.append(np.real(t))
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


'''