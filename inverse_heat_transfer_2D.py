"""
================================================================================
  2D INVERSE HEAT TRANSFER SOLVER  —  v4.2 (Real Experimental Data Integration)
  Glycerol-Water Cuvette  |  Conservative FDM  |  CuPy / Numba  |  scipy
================================================================================

USAGE
─────
  1. Fill in Section 1 (geometry, probe coords, h values, mixture VF).
  2. Point EXP_DATA_FILE to a CSV containing your real time-series data.
  3. Set USE_EXTERNAL_PROPERTIES = True — mixture_properties.py must be
     in the same folder.
  4. Run:  python inverse_heat_transfer_2d.py

Dependencies (core): numpy, scipy, matplotlib, numba, pandas
Optional (GPU):      cupy-cudaXX
Optional (props):    mixture_properties.py + its dependencies
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND SELECTION  (CuPy → GPU,  NumPy+Numba → CPU)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import cupy as xp
    _GPU = True
    _ = xp.array([0.0])          # warm-up CUDA context
    print("  [backend] CuPy detected  — GPU path active")
except ImportError:
    import numpy as xp
    _GPU = False
    print("  [backend] CuPy not found — CPU+Numba path active")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
from numba import njit, prange
import os as _os


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

USE_EXTERNAL_PROPERTIES  = True    # set False to use placeholder polynomials
VOLUME_FRACTION_GLYCEROL = 0.9     # glycerol volume fraction [0–1]

# ── Cuvette geometry ──────────────────────────────────────────────────────────
Lx = 0.012          # width  [m]  →  12 mm
Ly = 0.037          # height [m]  →  37 mm

dx = 1e-3           # spatial step x [m] (0.5 mm)
dy = 1e-3           # spatial step y [m] (0.5 mm)

# ── Temperatures & Time ───────────────────────────────────────────────────────
T_amb    = 25.5     # ambient air temperature [°C]
T_src_lb = T_amb + 0.5
T_src_ub = 150.0
t_total  = 180.0    # [s]

# ── Convective Enhancement & Boundaries ───────────────────────────────────────
CONVECTION_MULTIPLIER = 15.0  # Simulates natural convection fluid mixing
h_top   = 10.0      # [W/(m²·K)]  top wall
h_left  = 10.0      # [W/(m²·K)]  left wall
h_right = 10.0      # [W/(m²·K)]  right wall

# ── EXPERIMENTAL DATA INTEGRATION ─────────────────────────────────────────────
# Point this to your actual CSV file containing the time-series measurements
EXP_DATA_FILE = r"C:\Users\abdel\Desktop\data\my_experiment.csv"
EXP_TIME_COL  = "Time_s"        # Name of the time column in your CSV
EXP_TEMP_COL  = "Temperature"   # Name of the temperature column in your CSV

exp_probe = {
    'label' : 'Experimental probe (x0,y0)',
    'x'     : 0.006,    # [m]
    'y'     : 0.026,    # [m]
}

# ── Prediction points  (x1,y1)  (x2,y2) ──────────────────────────────────────
pred_points = [
    {'label': 'Prediction point 1 (x1,y1)', 'x': 0.006, 'y': 0.037},
    {'label': 'Prediction point 2 (x2,y2)', 'x': 0.006, 'y': 0.006},
]

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR        = r"C:\Users\abdel\Desktop\figures\figures"   # Windows path
SAVE_ANIMATION = True

_os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MATERIAL PROPERTY FUNCTIONS  (always plain NumPy)
# ══════════════════════════════════════════════════════════════════════════════

if USE_EXTERNAL_PROPERTIES:
    import importlib, sys, os
    _prop_dir = os.path.dirname(os.path.abspath(__file__))
    if _prop_dir not in sys.path:
        sys.path.insert(0, _prop_dir)
    _props = importlib.import_module("mixture_properties")
    print("  [props]   mixture_properties.py loaded")

    def k_func(T_arr: np.ndarray, vf: float = VOLUME_FRACTION_GLYCEROL) -> np.ndarray:
        shape  = np.asarray(T_arr).shape
        T_flat = np.asarray(T_arr, dtype=np.float64).ravel()
        out    = np.asarray(_props.k_mixture(np.full(T_flat.size, vf), T_flat), dtype=np.float64)
        return (out * CONVECTION_MULTIPLIER).reshape(shape)

    def rho_cp_func(T_arr: np.ndarray, vf: float = VOLUME_FRACTION_GLYCEROL) -> np.ndarray:
        shape  = np.asarray(T_arr).shape
        T_flat = np.asarray(T_arr, dtype=np.float64).ravel()
        vf_arr = np.full(T_flat.size, vf)
        rho    = np.asarray(_props.density_mixture(vf_arr, T_flat), dtype=np.float64)
        cp     = np.asarray(_props.c_p_mixture(vf_arr, T_flat),     dtype=np.float64)
        return (rho * cp).reshape(shape)

else:
    def k_func(T_arr, vf=VOLUME_FRACTION_GLYCEROL):
        T = np.asarray(T_arr, dtype=np.float64)
        return (0.38 + 1.5e-4 * T) * CONVECTION_MULTIPLIER

    def rho_cp_func(T_arr, vf=VOLUME_FRACTION_GLYCEROL):
        T   = np.asarray(T_arr, dtype=np.float64)
        rho = 1126.0 - 0.45 * T
        cp  = 3300.0 +  1.5 * T
        return rho * cp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GRID, STABILITY, TIME AXIS
# ══════════════════════════════════════════════════════════════════════════════

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

def _to_node(x, y):
    i = int(np.clip(round(x / dx), 1, Nx - 2))
    j = int(np.clip(round(y / dy), 1, Ny - 2))
    return i, j

exp_probe['i'], exp_probe['j'] = _to_node(exp_probe['x'], exp_probe['y'])
for pt in pred_points:
    pt['i'], pt['j'] = _to_node(pt['x'], pt['y'])

_T_test    = np.array([T_src_ub])
_alpha_max = float(k_func(_T_test)[0]) / float(rho_cp_func(_T_test)[0])
_dt_cfl    = 0.45 * (dx**2 * dy**2) / (2.0 * _alpha_max * (dx**2 + dy**2))
dt         = min(1.0, _dt_cfl)
Nt         = int(np.ceil(t_total / dt)) + 1
t_axis     = np.linspace(0.0, t_total, Nt)

r_dx2 = float(1.0 / dx**2)
r_dy2 = float(1.0 / dy**2)

print("=" * 68)
print("  2D INVERSE HEAT TRANSFER — Glycerol-Water Cuvette  (v4.2)")
print("=" * 68)
print(f"  Domain       : {Lx*1e3:.1f} mm × {Ly*1e3:.1f} mm")
print(f"  Grid         : {Nx} × {Ny} = {Nx*Ny:,} nodes  (dx=dy={dx*1e3:.2f} mm)")
print(f"  Stable dt    : {dt:.5f} s  |  Nt = {Nt:,}  |  t_total = {t_total:.0f} s")
print(f"  VF glycerol  : {VOLUME_FRACTION_GLYCEROL:.2f}")
print(f"  Convection   : k * {CONVECTION_MULTIPLIER} (Effective Fluid Mixing)\n")
p = exp_probe
print(f"  Experimental probe (x0,y0)")
print(f"    ({p['x']*1e3:.1f} mm, {p['y']*1e3:.1f} mm)  node ({p['i']},{p['j']})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPERIMENTAL REFERENCE CURVE (Real Data Loading)
# ══════════════════════════════════════════════════════════════════════════════

try:
    print(f"\n  [data] Loading experimental data from: {EXP_DATA_FILE}")
    _exp_df = pd.read_csv(EXP_DATA_FILE)
    
    # Extract raw arrays
    t_meas = _exp_df[EXP_TIME_COL].values
    T_meas = _exp_df[EXP_TEMP_COL].values
    
    # Interpolate measured data onto the simulation's fast time axis
    exp_probe['curve'] = np.interp(t_axis, t_meas, T_meas)
    
    # Extract actual initial and final temperatures for the plots
    exp_probe['T_initial'] = exp_probe['curve'][0]
    exp_probe['T_final']   = exp_probe['curve'][-1]
    
    print(f"  [data] Successfully loaded {len(t_meas)} data points.")
    print(f"  [data] Actual T_initial = {exp_probe['T_initial']:.2f} °C")
    print(f"  [data] Actual T_final   = {exp_probe['T_final']:.2f} °C\n")
    print("=" * 68 + "\n")
    
except Exception as e:
    print(f"\n  [ERROR] Could not load experimental data: {e}")
    print("  Please check the EXP_DATA_FILE path and column names in Section 1.")
    import sys; sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROPERTY LUTs  (CPU → GPU once at startup)
# ══════════════════════════════════════════════════════════════════════════════

_LUT_N    = 4000
_LUT_T_np = np.linspace(T_amb - 10.0, T_src_ub + 10.0, _LUT_N)
_LUT_k_np    = k_func(_LUT_T_np)                          # [W/(m·K)]
_LUT_rcP_np  = rho_cp_func(_LUT_T_np)                    # [J/(m³·K)]
_LUT_alpha_np = _LUT_k_np / _LUT_rcP_np                   # [m²/s]  (kept for CFL)

_xp_LUT_T    = xp.asarray(_LUT_T_np,    dtype=xp.float64)
_xp_LUT_k    = xp.asarray(_LUT_k_np,    dtype=xp.float64)
_xp_LUT_rcP  = xp.asarray(_LUT_rcP_np,  dtype=xp.float64)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NUMBA CPU KERNEL
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True, fastmath=True)
def _fdm_step_cpu(
    T_old:   np.ndarray, T_new:   np.ndarray,
    lut_T:   np.ndarray, lut_k:   np.ndarray, lut_rcP: np.ndarray,
    T_src:   float,      T_amb:   float,      dt:      float,
    dx:      float,      dy:      float,
    h_top:   float,      h_left:  float,      h_right: float,
    Nx:      int,        Ny:      int,
) -> None:
    def interp1(xq, xp_arr, fp_arr):
        n  = len(xp_arr) - 1
        lo = 0; hi = n
        while lo < hi - 1:
            mid = (lo + hi) >> 1
            if xp_arr[mid] <= xq: lo = mid
            else:                 hi = mid
        t = (xq - xp_arr[lo]) / (xp_arr[lo+1] - xp_arr[lo])
        return fp_arr[lo] + t * (fp_arr[lo+1] - fp_arr[lo])

    r_dx2 = 1.0 / (dx * dx); r_dy2 = 1.0 / (dy * dy)

    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            T_c  = T_old[i,   j]
            T_ip = T_old[i+1, j]; T_im = T_old[i-1, j]
            T_jp = T_old[i,   j+1]; T_jm = T_old[i,   j-1]

            k_c  = interp1(T_c,  lut_T, lut_k); k_ip = interp1(T_ip, lut_T, lut_k)
            k_im = interp1(T_im, lut_T, lut_k); k_jp = interp1(T_jp, lut_T, lut_k)
            k_jm = interp1(T_jm, lut_T, lut_k)

            k_xp = 2.0 * k_c * k_ip / (k_c + k_ip); k_xm = 2.0 * k_c * k_im / (k_c + k_im)
            k_yp = 2.0 * k_c * k_jp / (k_c + k_jp); k_ym = 2.0 * k_c * k_jm / (k_c + k_jm)

            div_flux = (k_xp*(T_ip - T_c) - k_xm*(T_c - T_im)) * r_dx2 \
                     + (k_yp*(T_jp - T_c) - k_ym*(T_c - T_jm)) * r_dy2

            rcp = interp1(T_c, lut_T, lut_rcP)
            T_new[i, j] = T_c + dt * div_flux / rcp

    for i in prange(Nx): T_new[i, 0] = T_src

    for i in prange(1, Nx - 1):
        j = Ny - 1
        T_c = T_old[i, j]; T_im = T_old[i-1, j]
        T_ip = T_old[i+1, j]; T_jm = T_old[i, j-1]

        k_c  = interp1(T_c,  lut_T, lut_k); k_ip = interp1(T_ip, lut_T, lut_k)
        k_im = interp1(T_im, lut_T, lut_k); k_jm = interp1(T_jm, lut_T, lut_k)

        k_xp = 2.0 * k_c * k_ip / (k_c + k_ip); k_xm = 2.0 * k_c * k_im / (k_c + k_im)
        k_ym = 2.0 * k_c * k_jm / (k_c + k_jm)

        T_ghost_top = T_jm - 2.0 * dy * h_top * (T_c - T_amb) / k_c
        k_yp_ghost  = k_c

        div_flux = (k_xp*(T_ip - T_c) - k_xm*(T_c - T_im)) * r_dx2 \
                 + (k_yp_ghost*(T_ghost_top - T_c) - k_ym*(T_c - T_jm)) * r_dy2

        rcp = interp1(T_c, lut_T, lut_rcP)
        T_new[i, j] = T_c + dt * div_flux / rcp

    for j in prange(1, Ny - 1):
        i = 0
        T_c = T_old[i, j]; T_ip = T_old[i+1, j]
        T_jp = T_old[i, j+1]; T_jm = T_old[i, j-1]

        k_c  = interp1(T_c,  lut_T, lut_k); k_ip = interp1(T_ip, lut_T, lut_k)
        k_jp = interp1(T_jp, lut_T, lut_k); k_jm = interp1(T_jm, lut_T, lut_k)

        k_xp = 2.0 * k_c * k_ip / (k_c + k_ip); k_yp = 2.0 * k_c * k_jp / (k_c + k_jp)
        k_ym = 2.0 * k_c * k_jm / (k_c + k_jm)

        T_ghost_left = T_ip - 2.0 * dx * h_left * (T_c - T_amb) / k_c
        k_xm_ghost   = k_c

        div_flux = (k_xp*(T_ip - T_c) - k_xm_ghost*(T_c - T_ghost_left)) * r_dx2 \
                 + (k_yp*(T_jp - T_c) - k_ym*(T_c - T_jm)) * r_dy2

        rcp = interp1(T_c, lut_T, lut_rcP)
        T_new[i, j] = T_c + dt * div_flux / rcp

    for j in prange(1, Ny - 1):
        i = Nx - 1
        T_c = T_old[i, j]; T_im = T_old[i-1, j]
        T_jp = T_old[i, j+1]; T_jm = T_old[i, j-1]

        k_c  = interp1(T_c,  lut_T, lut_k); k_im = interp1(T_im, lut_T, lut_k)
        k_jp = interp1(T_jp, lut_T, lut_k); k_jm = interp1(T_jm, lut_T, lut_k)

        k_xm = 2.0 * k_c * k_im / (k_c + k_im); k_yp = 2.0 * k_c * k_jp / (k_c + k_jp)
        k_ym = 2.0 * k_c * k_jm / (k_c + k_jm)

        T_ghost_right = T_im - 2.0 * dx * h_right * (T_c - T_amb) / k_c
        k_xp_ghost    = k_c

        div_flux = (k_xp_ghost*(T_ghost_right - T_c) - k_xm*(T_c - T_im)) * r_dx2 \
                 + (k_yp*(T_jp - T_c) - k_ym*(T_c - T_jm)) * r_dy2

        rcp = interp1(T_c, lut_T, lut_rcP)
        T_new[i, j] = T_c + dt * div_flux / rcp

    for _corner in range(1):
        i, j = 0, Ny-1; T_new[i, j] = 0.5 * (T_new[1, j] + T_new[i, j-1])
        i, j = Nx-1, Ny-1; T_new[i, j] = 0.5 * (T_new[Nx-2, j] + T_new[i, j-1])

print("  [numba]   Compiling JIT kernel … ", end='', flush=True)
_dummy_T = np.full((Nx, Ny), float(T_amb))
_dummy_out = _dummy_T.copy()
_fdm_step_cpu(_dummy_T, _dummy_out, _LUT_T_np, _LUT_k_np, _LUT_rcP_np, float(T_src_lb), T_amb, dt, dx, dy, h_top, h_left, h_right, Nx, Ny)
print("done")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GPU CONSERVATIVE FLUX STEP  (CuPy vectorised)
# ══════════════════════════════════════════════════════════════════════════════

def _fdm_step_gpu(T, T_out, T_src_val):
    def _c(a):     return xp.ascontiguousarray(a)
    def k_lut(a):  return xp.interp(_c(a), _xp_LUT_T, _xp_LUT_k)
    def rc_lut(a): return xp.interp(_c(a), _xp_LUT_T, _xp_LUT_rcP)

    kT = k_lut(T)
    kA = kT[:-1, :]; kB = kT[1:,  :];  k_xface = 2.0*kA*kB/(kA+kB)
    kA = kT[:, :-1]; kB = kT[:, 1:] ;  k_yface = 2.0*kA*kB/(kA+kB)

    flux_x = (  k_xface[1:,  1:-1] * (T[2:,   1:-1] - T[1:-1, 1:-1])
              - k_xface[:-1, 1:-1] * (T[1:-1,  1:-1] - T[:-2,  1:-1]) ) * r_dx2

    flux_y = (  k_yface[1:-1, 1:] * (T[1:-1, 2:]  - T[1:-1, 1:-1])
              - k_yface[1:-1,:-1] * (T[1:-1, 1:-1] - T[1:-1, :-2])  ) * r_dy2

    rcp_int = rc_lut(T[1:-1, 1:-1])

    xp.copyto(T_out, T)
    T_out[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (flux_x + flux_y) / rcp_int

    T_out[:, 0] = T_src_val

    j = Ny - 1
    T_c = T[:, j]; T_jm = T[:, j-1]; k_c = kT[:, j]; k_jm = kT[:, j-1]
    T_g_top = T_jm - (2.0*dy*h_top/k_c)*(T_c - T_amb)
    k_ym_w  = 2.0*k_c*k_jm/(k_c+k_jm)
    fx_t = (  k_xface[1:,  j]*(T[2:,  j] - T[1:-1, j])
            - k_xface[:-1, j]*(T[1:-1, j] - T[:-2,  j]) ) * r_dx2
    fy_t = ( k_c[1:-1]*(T_g_top[1:-1]-T_c[1:-1])
           - k_ym_w[1:-1]*(T_c[1:-1]-T_jm[1:-1]) ) * r_dy2
    T_out[1:-1, j] = T_c[1:-1] + dt*(fx_t+fy_t)/rc_lut(T_c[1:-1])

    i = 0
    T_c = T[i, :]; T_ip = T[i+1, :]; k_c = kT[i, :]; k_ip = kT[i+1, :]
    T_g_l = T_ip - (2.0*dx*h_left/k_c)*(T_c - T_amb)
    k_xp_w = 2.0*k_c*k_ip/(k_c+k_ip)
    fy_l = (  k_yface[i, 1:]*(T[i, 2:] -T[i, 1:-1])
            - k_yface[i,:-1]*(T[i, 1:-1]-T[i, :-2]) ) * r_dy2
    fx_l = ( k_xp_w[1:-1]*(T_ip[1:-1]-T_c[1:-1])
           - k_c[1:-1]*(T_c[1:-1]-T_g_l[1:-1]) ) * r_dx2
    T_out[i, 1:-1] = T_c[1:-1] + dt*(fx_l+fy_l)/rc_lut(T_c[1:-1])

    i = Nx - 1
    T_c = T[i, :]; T_im = T[i-1, :]; k_c = kT[i, :]; k_im = kT[i-1, :]
    T_g_r = T_im - (2.0*dx*h_right/k_c)*(T_c - T_amb)
    k_xm_w = 2.0*k_c*k_im/(k_c+k_im)
    fy_r = (  k_yface[i, 1:]*(T[i, 2:] -T[i, 1:-1])
            - k_yface[i,:-1]*(T[i, 1:-1]-T[i, :-2]) ) * r_dy2
    fx_r = ( k_c[1:-1]*(T_g_r[1:-1]-T_c[1:-1])
           - k_xm_w[1:-1]*(T_c[1:-1]-T_im[1:-1]) ) * r_dx2
    T_out[i, 1:-1] = T_c[1:-1] + dt*(fx_r+fy_r)/rc_lut(T_c[1:-1])

    T_out[0,    Ny-1] = 0.5*(T_out[1,    Ny-1] + T_out[0,    Ny-2])
    T_out[Nx-1, Ny-1] = 0.5*(T_out[Nx-2, Ny-1] + T_out[Nx-1, Ny-2])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FORWARD SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_forward_simulation(T_src: float, store_fields: bool = False):
    T_src_f = float(T_src)

    initial_liquid_temp = float(exp_probe['T_initial'])

    if _GPU:
        buf_A = xp.full((Nx, Ny), initial_liquid_temp, dtype=xp.float64)
        buf_B = xp.empty((Nx, Ny),               dtype=xp.float64)
    else:
        buf_A = np.full((Nx, Ny), initial_liquid_temp, dtype=np.float64)
        buf_B = np.empty((Nx, Ny),               dtype=np.float64)

    buf_A[:, 0] = T_src_f

    sim_probe   = np.empty(Nt, dtype=np.float64)
    pred_curves = [np.empty(Nt, dtype=np.float64) for _ in pred_points]

    sim_probe[0] = float(buf_A[exp_probe['i'], exp_probe['j']])
    for kp, pt in enumerate(pred_points):
        pred_curves[kp][0] = float(buf_A[pt['i'], pt['j']])

    if store_fields:
        fields = np.empty((Nt, Nx, Ny), dtype=np.float32)
        fields[0] = (xp.asnumpy(buf_A) if _GPU else buf_A).astype(np.float32)

    T_cur = buf_A
    T_nxt = buf_B

    for n in range(1, Nt):
        if _GPU:
            _fdm_step_gpu(T_cur, T_nxt, T_src_f)
        else:
            _fdm_step_cpu(
                T_cur, T_nxt,
                _LUT_T_np, _LUT_k_np, _LUT_rcP_np,
                T_src_f, T_amb, dt, dx, dy,
                h_top, h_left, h_right, Nx, Ny
            )

        T_cur, T_nxt = T_nxt, T_cur

        sim_probe[n] = float(T_cur[exp_probe['i'], exp_probe['j']])
        for kp, pt in enumerate(pred_points):
            pred_curves[kp][n] = float(T_cur[pt['i'], pt['j']])

        if store_fields:
            fields[n] = (xp.asnumpy(T_cur) if _GPU else T_cur).astype(np.float32)

    if store_fields:
        return sim_probe, pred_curves, fields
    return sim_probe, pred_curves

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

_iter_log = {"n": 0, "history": []}

def objective(T_src: float) -> float:
    _iter_log["n"] += 1
    sim_probe, _ = run_forward_simulation(float(T_src), store_fields=False)
    sse = float(np.sum((sim_probe - exp_probe['curve'])**2))
    _iter_log["history"].append((float(T_src), sse))
    print(f"  iter {_iter_log['n']:>3d}  |  "
          f"T_src = {T_src:8.3f} °C  |  SSE = {sse:.4f} °C²")
    return sse


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

print("  ──────────────────────────────────────────────────────")
print("  Optimisation  (minimize_scalar / Brent, bounded)      ")
print(f"  Search range  : [{T_src_lb:.1f}, {T_src_ub:.1f}] °C")
print("  ──────────────────────────────────────────────────────\n")

opt_result = minimize_scalar(
    objective,
    bounds=(T_src_lb, T_src_ub),
    method='bounded',
    options={'xatol': 0.02, 'maxiter': 80}
)

T_src_opt = float(opt_result.x)
sse_opt   = float(opt_result.fun)
rmse_opt  = float(np.sqrt(sse_opt / Nt))

print(f"\n  Converged in {_iter_log['n']} iterations")
print("=" * 68)
print("  RESULT")
print("=" * 68)
print(f"  Recovered plaque temperature : {T_src_opt:.3f} °C")
print(f"  SSE  at (x0,y0)              : {sse_opt:.4f} °C²")
print(f"  RMSE at (x0,y0)              : {rmse_opt:.4f} °C")
print("=" * 68 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FINAL SIMULATION & DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"  Running final simulation at T_src = {T_src_opt:.3f} °C …\n")
final_sim, final_preds, final_fields = run_forward_simulation(
    T_src_opt, store_fields=True)

for kp, pt in enumerate(pred_points):
    print(f"  {pt['label']}")
    print(f"    T(t=0)        = {final_preds[kp][0]:.3f} °C")
    print(f"    T(t={t_total:.0f} s) = {final_preds[kp][-1]:.3f} °C")
print()

# ── EXPORT TO PANDAS DATAFRAME ──
print("  Bundling data into pandas DataFrame...")
df_temp = pd.DataFrame({
    'Time_s': t_axis,
    'T_Probe_x0y0_degC': final_sim
})

for kp, pt in enumerate(pred_points):
    col_suffix = pt['label'].split(' ')[2] 
    df_temp[f"T_Pred_Pt{col_suffix}_degC"] = final_preds[kp]

csv_path = _os.path.join(OUT_DIR, "recovered_temperatures.csv")
df_temp.to_csv(csv_path, index=False)

print(f"  [Data Export] Successfully saved full time-series to -> {csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name):
    path = _os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")

_stride   = max(1, Nt // 600)
t_plt     = t_axis[::_stride]
_cpred    = ['crimson', 'royalblue']
_mpred    = ['^', 's']

fig1 = plt.figure(figsize=(17, 5))
fig1.suptitle(
    f"Inverse Problem — Recovered plaque temperature: {T_src_opt:.2f} °C  "
    f"(VF={VOLUME_FRACTION_GLYCEROL:.2f}  h={h_top}/{h_left}/{h_right} W/m²K)",
    fontsize=12, fontweight='bold')
gs  = GridSpec(1, 3, figure=fig1, wspace=0.35)
ax1 = fig1.add_subplot(gs[0])
ax2 = fig1.add_subplot(gs[1])
ax3 = fig1.add_subplot(gs[2])

ref_plt = exp_probe['curve'][::_stride]
sim_plt = final_sim[::_stride]

ax1.plot(t_plt, ref_plt, '-',  lw=2.5, color='dimgray', label='Real Experimental Data')
ax1.plot(t_plt, sim_plt, '--', lw=2.0, color='darkorange', label='Simulation at (x0,y0)')
ax1.scatter([0, t_total], [exp_probe['T_initial'], exp_probe['T_final']], color='black', s=80, zorder=6, label='T_initial / T_final')
ax1.axhline(exp_probe['T_initial'], color='black', ls=':', lw=0.8, alpha=0.4)
ax1.axhline(exp_probe['T_final'],   color='black', ls=':', lw=0.8, alpha=0.4)
ax1.set_xlabel("Time  [s]");  ax1.set_ylabel("Temperature  [°C]")
ax1.set_title(f"Fit at exp. probe\n({exp_probe['x']*1e3:.0f} mm, {exp_probe['y']*1e3:.0f} mm)")
ax1.legend(fontsize=8);  ax1.grid(True, ls='--', alpha=0.4)

res_plt = sim_plt - ref_plt
ax2.plot(t_plt, res_plt, '-', lw=1.5, color='darkorange')
ax2.fill_between(t_plt, res_plt, alpha=0.18, color='darkorange')
ax2.axhline(0, color='gray', ls='--', lw=0.9)
ax2.set_xlabel("Time  [s]");  ax2.set_ylabel("T_sim − T_ref  [°C]")
ax2.set_title(f"Residuals at (x0,y0)\nRMSE = {rmse_opt:.3f} °C")
ax2.grid(True, ls='--', alpha=0.4)

hist_T   = [h[0] for h in _iter_log["history"]]
hist_sse = [h[1] for h in _iter_log["history"]]
sc = ax3.scatter(hist_T, hist_sse, c=range(len(hist_T)), cmap='plasma', s=65, zorder=3)
ax3.axvline(T_src_opt, color='crimson', ls='--', lw=1.5, label=f'Optimum {T_src_opt:.1f} °C')
plt.colorbar(sc, ax=ax3, label='Iteration')
ax3.set_xlabel("Candidate T_src  [°C]");  ax3.set_ylabel("SSE  [°C²]")
ax3.set_title("Optimiser search path");  ax3.legend(fontsize=9)
ax3.grid(True, ls='--', alpha=0.4)

fig1.tight_layout()
_savefig(fig1, "inverse_fit.png")

fig2, ax_p = plt.subplots(figsize=(8, 5))
fig2.suptitle(
    f"Predicted T(t) from T_src = {T_src_opt:.2f} °C\n"
    f"Robin BCs: h = {h_top} W/(m²·K)",
    fontsize=12, fontweight='bold')

ax_p.plot(t_plt, ref_plt, '-', lw=2.0, color='dimgray', alpha=0.7, label=f"Exp. probe (x0,y0) = ({exp_probe['x']*1e3:.0f},{exp_probe['y']*1e3:.0f}) mm")
for kp, pt in enumerate(pred_points):
    pp = final_preds[kp][::_stride]
    ax_p.plot(t_plt, pp, '-', lw=2.2, color=_cpred[kp], label=f"{pt['label']}  ({pt['x']*1e3:.0f},{pt['y']*1e3:.0f}) mm")
    ax_p.scatter([0, t_total], [final_preds[kp][0], final_preds[kp][-1]], color=_cpred[kp], s=60, zorder=5)

ax_p.set_xlabel("Time  [s]");  ax_p.set_ylabel("Temperature  [°C]")
ax_p.set_title("Physics-only temperature predictions")
ax_p.legend(fontsize=9);  ax_p.grid(True, ls='--', alpha=0.4)
fig2.tight_layout()
_savefig(fig2, "predictions.png")

snap_idx  = [0, Nt//4, Nt//2, Nt-1]
T_vmin    = T_amb - 0.5
T_vmax    = float(final_fields.max())
extent_mm = [0, Lx*1e3, 0, Ly*1e3]

fig3, axes3 = plt.subplots(1, 4, figsize=(17, 6))
fig3.suptitle(
    f"Full 2D Temperature Field T(x,y,t)  |  T_src = {T_src_opt:.2f} °C  "
    f"|  Robin h = {h_top} W/(m²·K)",
    fontsize=12, fontweight='bold')

for ax, idx in zip(axes3, snap_idx):
    img = final_fields[idx].T[::-1, :]
    im  = ax.imshow(img, origin='upper', extent=extent_mm, aspect='auto', cmap='inferno', vmin=T_vmin, vmax=T_vmax)
    ax.plot(exp_probe['x']*1e3, exp_probe['y']*1e3, 'w*', ms=11, zorder=6, label='(x0,y0)')
    for kp, pt in enumerate(pred_points):
        ax.plot(pt['x']*1e3, pt['y']*1e3, _mpred[kp], ms=8, color=_cpred[kp], markeredgecolor='white', markeredgewidth=0.7, label=f"pred {kp+1}", zorder=5)
    ax.axhline(0.5, color='white', lw=0.9, ls='--', alpha=0.3)
    ax.text(Lx*1e3/2, 0.9, f'Plaque {T_src_opt:.1f} °C', ha='center', va='bottom', color='white', fontsize=7)
    ax.set_title(f"t = {t_axis[idx]:.0f} s")
    ax.set_xlabel("x  [mm]");  ax.set_ylabel("y  [mm]")
    ax.legend(fontsize=7, loc='upper right')
    plt.colorbar(im, ax=ax, label="T [°C]", fraction=0.046, pad=0.04)

fig3.tight_layout()
_savefig(fig3, "heatmap_snapshots.png")

if SAVE_ANIMATION:
    print("\n  Building animation …")
    _a_stride = max(1, Nt // 90)
    _a_frames = list(range(0, Nt, _a_stride))
    fig4, ax4  = plt.subplots(figsize=(4, 7))
    im4 = ax4.imshow(final_fields[0].T[::-1, :], origin='upper', extent=extent_mm, aspect='auto', cmap='inferno', vmin=T_vmin, vmax=T_vmax)
    ax4.plot(exp_probe['x']*1e3, exp_probe['y']*1e3, 'w*', ms=11, label='(x0,y0)')
    for kp, pt in enumerate(pred_points):
        ax4.plot(pt['x']*1e3, pt['y']*1e3, _mpred[kp], ms=8, color=_cpred[kp], markeredgecolor='white', label=f"pred {kp+1}")
    fig4.colorbar(im4, ax=ax4, label="T [°C]")
    ax4.set_xlabel("x  [mm]");  ax4.set_ylabel("y  [mm]");  ax4.legend(fontsize=9)

    def _update(fi):
        im4.set_data(final_fields[fi].T[::-1, :])
        ax4.set_title(f"t = {t_axis[fi]:.1f} s")
        return [im4]

    ani = animation.FuncAnimation(fig4, _update, frames=_a_frames, interval=80, blit=True)
    gif_path = _os.path.join(OUT_DIR, "temperature_evolution.gif")
    ani.save(gif_path, writer='pillow', fps=12, dpi=100)
    print(f"  Saved -> {gif_path}")
    plt.close(fig4)

print("\n  Done.")
