import glob
import os
from pathlib import Path
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from scipy import stats

# ── GPU setup ──────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    # Quick check that a GPU is actually available
    cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    print("✅ GPU detected — running on GPU (CuPy).")
except Exception:
    import numpy as cp          # CuPy-compatible fallback: cp becomes numpy
    GPU_AVAILABLE = False
    print("⚠️  No GPU / CuPy not installed — falling back to CPU (numpy).")

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# ── Dark frame ─────────────────────────────────────────────────────────────────
def get_master_dark(dark_folder_path, target_shape=(240, 320)):
    """Reads all dark TIFFs, averages, and resizes to target_shape."""
    print("\n--- Processing Dark Frames (Fond Noir) ---")
    search_pattern = os.path.join(dark_folder_path, "*.tiff")
    dark_files = glob.glob(search_pattern)

    if not dark_files:
        print("❌ No dark files found! Returning 0 (no subtraction).")
        return 0.0

    print(f"Found {len(dark_files)} dark images. Averaging...")
    dark_stack = np.array([tiff.imread(f).astype(float) for f in dark_files])
    master_dark = np.mean(dark_stack, axis=0)

    print(f"Original Dark shape: {master_dark.shape}. Resizing to {target_shape}...")
    target_height, target_width = target_shape
    master_dark_resized = cv2.resize(
        master_dark,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    print(f"✅ Master Dark created and resized to: {master_dark_resized.shape}")
    return master_dark_resized


# ── GPU-accelerated correlation ────────────────────────────────────────────────
def compute_g_tau(images_gpu, max_tau):
    """
    Vectorized g2(tau)-1 using CuPy (or numpy as fallback).

    images_gpu : cp.ndarray of shape (N, H, W) — already on GPU
    max_tau    : int, maximum lag in frames
    Returns    : cp.ndarray of length max_tau (on GPU)
    """
    N = len(images_gpu)
    g_tau = cp.zeros(max_tau)

    for tau in range(max_tau):
        im1 = images_gpu[:N - tau]                          # (N-tau, H, W)
        im2 = images_gpu[tau:]                              # (N-tau, H, W)

        num = cp.mean(im1 * im2, axis=(1, 2))               # spatial mean per pair
        den = cp.mean(im1, axis=(1, 2)) * cp.mean(im2, axis=(1, 2))

        g_tau[tau] = cp.mean(num / den) - 1

    return g_tau


def to_numpy(arr):
    """Safely convert a CuPy or numpy array to numpy."""
    if GPU_AVAILABLE:
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ── Paths ──────────────────────────────────────────────────────────────────────
dark_folder_path = r"C:\Users\abdel\Desktop\etudes\PT_Speckle\manip3\SP_30\FN"
folder = Path(r"C:\Users\abdel\Desktop\etudes\PT_Speckle\manip3\SP_30\laser")

# Peek at laser images first to determine the minimum resolution,
# so the dark frame is resized to the same target from the start.
_preview_files = sorted(p for p in folder.iterdir() if p.is_file())
_shapes        = [tiff.imread(f).shape for f in _preview_files]
_min_h         = min(s[0] for s in _shapes)
_min_w         = min(s[1] for s in _shapes)

bg_im = get_master_dark(dark_folder_path, target_shape=(_min_h, _min_w))

files = _preview_files          # already sorted above
N     = len(files)
print(f"\nFound {N} images in {folder}")

# ── Auto-detect minimum resolution across all images ──────────────────────────
shapes       = _shapes          # already read above
min_h, min_w = _min_h, _min_w
target_shape = (min_h, min_w)

unique_shapes = set(tuple(s) for s in shapes)
if len(unique_shapes) == 1:
    print(f"✅ All images are the same size {shapes[0]} — no scaling needed.")
else:
    print(f"⚠️  Mixed resolutions detected: {unique_shapes}")
    print(f"   Scaling all images down to smallest resolution: {target_shape}")

def resize_if_needed(img, target_h, target_w):
    """Resize image to (target_h, target_w) only if it is larger."""
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

# Load images on CPU first, then upload the whole stack to GPU in one transfer
print("Loading images and uploading to GPU...")
images_cpu = np.array([
    resize_if_needed(tiff.imread(files[i]).astype(np.float64), min_h, min_w) - bg_im
    for i in range(N)
])  # shape (N, H, W)
print(f"Image array shape: {images_cpu.shape}")

images_gpu = cp.asarray(images_cpu)   # single host→device transfer
print(f"✅ Images uploaded to {'GPU VRAM' if GPU_AVAILABLE else 'CPU RAM (fallback)'}.")

# ── Main correlation curve ─────────────────────────────────────────────────────
max_tau = 50
print(f"\nComputing g2(tau) for max_tau={max_tau}...")
g_gpu  = compute_g_tau(images_gpu, max_tau)
g      = to_numpy(g_gpu)               # bring result back to CPU for plotting

# Subtract baseline from the tail (last 10 points)
baseline = np.mean(g[-10:])
g = g - baseline
# Fit region
ibegin, iend = 0, 40
x = np.arange(0, max_tau) / 120       # time in seconds (120 fps)

x_fit     = x[ibegin:iend]
log_g_fit = np.log(g[ibegin:iend])

regress = stats.linregress(x_fit, log_g_fit)
slope   = regress.slope
tau_c   = -1 / slope

print(f"\nEstimated slope (1/Tau_c): {slope:.4f} 1/s")
print(f"Estimated correlation time Tau_c (main curve): {tau_c:.4f} s")

y_reg_full = slope * x + regress.intercept

# ── Robustness loop — compute g_full once on GPU, then slice ──────────────────
max_frames_list = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
baseline_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

print("\n--- Robustness analysis over different max_frames ---")

# Single GPU computation up to the largest lag needed
g_full_gpu = compute_g_tau(images_gpu, max(max_frames_list))
g_full     = to_numpy(g_full_gpu)      # one device→host transfer for all slices

Tau_list = []

for i, max_frame in enumerate(max_frames_list):
    g1 = g_full[:max_frame].copy()

    baseline1 = np.mean(g1[-baseline_values[i]:])
    g1 = g1 - baseline1

    x1 = np.arange(0, max_frame) / 120

    ibegin_frame, iend_frame = 10, 40
    if iend_frame >= max_frame:
        print(f"Warning: iend_frame={iend_frame} >= max_frame={max_frame}; skipping fit.")
        continue

    x1_fit     = x1[ibegin_frame:iend_frame]
    log_g1_fit = np.log(g1[ibegin_frame:iend_frame])

    regress1 = stats.linregress(x1_fit, log_g1_fit)
    slope1   = regress1.slope
    tau_c1   = -1 / slope1
    Tau_list.append(tau_c1)

    print(f"max_frame={max_frame}, baseline_len={baseline_values[i]}, Tau_c = {tau_c1:.4f} s")

print()

T = 26 + 273.15  # Convert °C to K
n =  1.4619
eta = 0.1937
theta = np.pi / 2
lamda = 532 * 10**(-9)
kB = 1.380649e-23   # J/K

# Uncertainty 
dT = 1
d_eta = eta * 0.05  # 5% uncertainty in viscosity
d_n = 0.0005
d_theta = np.pi / 180


# ── Statistics ─────────────────────────────────────────────────────────────────
if Tau_list:
    d_tau = np.std(Tau_list)
    print(f"Mean Tau_c over different max_frames: {np.mean(Tau_list):.4f} s")
    print(f"Std Dev of Tau_c:                     {np.std(Tau_list):.4f} s")
    tau = np.mean(Tau_list)
    q = (4 * np.pi * n / lamda) * np.sin(theta / 2)
    D = 1 / (2 * tau * q **2)
    a = kB * T / (6 * np.pi * eta * D)
    d_q = 4 * np.pi / lamda * np.sqrt(np.sin(theta/2)**2 * d_n**2 + (n**2 * np.cos(theta/2)**2 * d_theta**2) / 4)
    d_D = np.sqrt((1/(2*tau**2*q**2)*d_tau)**2 + ((1/(tau*q**3))*d_q)**2)
    d_a = np.sqrt((kB*T/(6*np.pi*eta*D**2)*d_D)**2 + (kB*T/(6*np.pi*eta**2*D)*d_eta)**2 + (kB*dT/(6*np.pi*eta*D))**2)
    print(f"Scattering vector q: {q:.4e} 1/m")
    print(f"Estimated diffusion coefficient D: {D:.4e} m^2/s")
    print(f"Uncertainty in D: {d_D:.4e} m^2/s")
    print(f"Estimated radius a: {a:.4e} m")
    print(f"Uncertainty in a: {d_a:.4e} m")
else:
    print("No valid fits were computed (probably max_frame too small).")

# ── Plotting (always on CPU / matplotlib) ─────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(x[:41], g[:41], "-o", label=r"$g_2(\tau)-1$")
axs[0].axhline(0, color="k", ls="--", alpha=0.5)
axs[0].set_xlabel(r"Time Lag $\tau$ (s)")
axs[0].set_ylabel(r"$g_2(\tau) - 1$")
axs[0].set_title("Intensity Correlation Function")
axs[0].legend()

axs[1].plot(x[:40], np.log(g[:40]), "-", label=r"$\log(g_2(\tau)-1)$")
axs[1].plot(x[:40], y_reg_full[:40], "--", color="r", label=f"Fit (slope={slope:.2f})")
axs[1].axvline(x[ibegin],   color="gray", ls=":", alpha=0.5)
axs[1].axvline(x[iend - 1], color="gray", ls=":", alpha=0.5)
axs[1].set_xlabel(r"Time Lag $\tau$ (s)")
axs[1].set_ylabel(r"$\log(g_2(\tau)-1)$")
axs[1].set_title("Log Intensity Correlation (linear fit)")
axs[1].legend()

plt.tight_layout()
plt.show()
