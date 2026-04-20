import matplotlib
matplotlib.use("Agg")                      # must come first — no GUI, no tkinter threads
import matplotlib as mpl
import matplotlib.pyplot as plt

import mixture_properties as gly
import glob
import os
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from scipy import stats

# ── GPU setup ──────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    print("✅ GPU detected — running on GPU (CuPy).")
except Exception:
    import numpy as cp          # type: ignore[no-redef]
    GPU_AVAILABLE = False
    print("⚠️  No GPU / CuPy not installed — falling back to CPU (numpy).")

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    # --- Shared dark folder (one for all experiments) -------------------------
    dark_folder: str = r"C:\Users\abdel\Desktop\etudes\manip4\FN_90"

    # --- Experiments: (laser_folder, temperature_K) ---------------------------
    experiments: list = field(default_factory=lambda: [
        (r"C:\Users\abdel\Desktop\etudes\manip4\Laser_90_al25_25C", 26 + 273.15),
        (r"C:\Users\abdel\Desktop\etudes\manip4\Laser_90_al25_29", 29 + 273.15),
        (r"C:\Users\abdel\Desktop\etudes\manip4\Laser_90_al25_33", 33 + 273.15),
        (r"C:\Users\abdel\Desktop\etudes\manip4\Laser_90_al25_38", 38 + 273.15),
    ])

    # --- Sample properties ----------------------------------------------------
    glycerol_vol_pct: float = 90.0   # glycerol volume % — converted to fraction internally
    a:               float = 9.3e-08   # known hydrodynamic radius (m)

    # --- Acquisition ----------------------------------------------------------
    fps: float = 120.0

    # --- Correlation ----------------------------------------------------------
    max_tau: int = 50

    # --- Robustness sweep -----------------------------------------------------
    max_frames_list: list = field(default_factory=lambda: list(range(50, 100, 5)))
    baseline_values: list = field(default_factory=lambda: list(range(5, 55, 5)))

    # --- Physical constants ---------------------------------------------------
    theta: float = np.pi / 2    # scattering angle (rad)
    lamda: float = 532e-9       # laser wavelength (m)
    kB:    float = 1.380649e-23 # Boltzmann constant (J/K)

    # --- Output folder --------------------------------------------------------
    out_dir: str = r"C:\Users\abdel\Desktop\D_reults\Stat_sys"


cfg = Config()


# ── Per-temperature physical parameters from mixture_properties ────────────────
def get_sample_params(c: Config, T_C: float) -> tuple[float, float]:
    vol_frac = c.glycerol_vol_pct / 100.0   # 90.0 % → 0.9
    lamda_nm = c.lamda * 1e9                # 532e-9 m → 532.0 nm

    n_arr = gly.n_mixture(vol_frac, T_C, lamda=lamda_nm)
    n     = float(np.atleast_1d(n_arr)[0])

    nabla = float(gly.dyn_visc_mixture(vol_frac, T_C))

    return n, nabla


# ── Dynamic fit-window helper ──────────────────────────────────────────────────
def find_valid_fit_range(g: np.ndarray, ibegin: int = 0) -> tuple[int, int]:
    """
    Return (ibegin, iend) so that g[ibegin:iend] are all strictly positive.
    Stops just before the first index where g <= 0.
    Raises ValueError if g[ibegin] itself is non-positive.
    """
    bad  = np.where(g[ibegin:] <= 0)[0]
    iend = ibegin + int(bad[0]) if bad.size > 0 else len(g)

    if iend <= ibegin:
        raise ValueError(
            f"g[{ibegin}] = {g[ibegin]:.4g} <= 0 — no positive values to fit."
        )
    return ibegin, iend


# ── Dark frame ─────────────────────────────────────────────────────────────────
def get_master_dark(
    dark_folder_path: str,
    target_shape: tuple[int, int],
) -> np.ndarray:
    print("\n--- Processing Dark Frames ---")
    dark_files = glob.glob(os.path.join(dark_folder_path, "*.tiff"))

    if not dark_files:
        print("  ❌ No dark files found — returning zero array.")
        return np.zeros(target_shape, dtype=np.float64)

    print(f"  Found {len(dark_files)} dark images — averaging in parallel…")

    def _load(f: str) -> np.ndarray:
        return tiff.imread(f).astype(np.float64)

    with ThreadPoolExecutor() as pool:
        dark_stack = np.array(list(pool.map(_load, dark_files)))

    master_dark = dark_stack.mean(axis=0)
    h, w = target_shape
    master_dark_resized = cv2.resize(master_dark, (w, h), interpolation=cv2.INTER_AREA)
    print(f"  ✅ Master dark shape: {master_dark_resized.shape}")
    return master_dark_resized


# ── Vectorised GPU correlation ─────────────────────────────────────────────────
def compute_g_tau(images_gpu: "cp.ndarray", max_tau: int) -> "cp.ndarray":
    N           = images_gpu.shape[0]
    frame_means = images_gpu.mean(axis=(1, 2))
    g_tau       = cp.empty(max_tau, dtype=cp.float64)

    for tau in range(max_tau):
        n_pairs    = N - tau
        im1        = images_gpu[:n_pairs]
        im2        = images_gpu[tau: tau + n_pairs]
        num        = (im1 * im2).mean(axis=(1, 2))
        den        = frame_means[:n_pairs] * frame_means[tau: tau + n_pairs]
        g_tau[tau] = (num / den).mean() - 1.0

    return g_tau


def compute_c_t_tau(images_gpu: "cp.ndarray", tau: int) -> "cp.ndarray":
    N       = images_gpu.shape[0]
    n_pairs = N - tau
    im1     = images_gpu[:n_pairs]
    im2     = images_gpu[tau: tau + n_pairs]
    num     = (im1 * im2).mean(axis=(1, 2))
    den     = im1.mean(axis=(1, 2)) * im2.mean(axis=(1, 2))
    return num / den - 1.0


def to_numpy(arr: "cp.ndarray") -> np.ndarray:
    return cp.asnumpy(arr) if GPU_AVAILABLE else np.asarray(arr)


# ── Parallel image loader ──────────────────────────────────────────────────────
def load_images_parallel(
    files: list[Path],
    min_h: int,
    min_w: int,
    bg: np.ndarray,
) -> np.ndarray:
    def _load_one(path: Path) -> np.ndarray:
        img = tiff.imread(path).astype(np.float64)
        h, w = img.shape[:2]
        if h != min_h or w != min_w:
            img = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
        return img - bg

    print(f"  Loading {len(files)} images in parallel…")
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_load_one, files))

    return np.array(results, dtype=np.float64)


# ── Physical quantities ────────────────────────────────────────────────────────
def compute_physics(T: float, n: float, nabla: float, c: Config) -> dict:
    """
    Compute q and theoretical D from known a (Stokes-Einstein).
    """
    q = (4 * np.pi * n / c.lamda) * np.sin(c.theta / 2)
    D_theo = c.kB * T / (6.0 * np.pi * nabla * c.a)
    return {"q": q, "D_theo": D_theo, "a": c.a}


# ── Linear fit on log(g) ──────────────────────────────────────────────────────
def fit_log_slope(
    g: np.ndarray,
    x: np.ndarray,
    ibegin: int,
    iend: int,
) -> tuple[float, float, float]:
    result = stats.linregress(x[ibegin:iend], np.log(g[ibegin:iend]))
    tau_c  = -1.0 / result.slope
    return result.slope, result.intercept, tau_c


# ══════════════════════════════════════════════════════════════════════════════
# Per-experiment pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(
    laser_folder: str,
    T: float,       # temperature in K
    c: Config,
) -> dict | None:
    """
    Run the full DLS pipeline for a single temperature T (Kelvin).
    Returns a dict {T, T_C, n, nabla, tau_c, q, D_theo, D_exp, a} or None on failure.
    """
    T_C = T - 273.15
    print(f"\n{'='*60}")
    print(f"  Temperature : {T:.2f} K  ({T_C:.1f} °C)")
    print(f"{'='*60}")

    # --- Temperature-dependent sample parameters ------------------------------
    n, nabla = get_sample_params(c, T_C)
    print(f"  Refractive index  n     = {n:.5f}")
    print(f"  Dynamic viscosity nabla = {nabla:.4e} Pa·s")

    # ── 1. Discover files ──────────────────────────────────────────────────────
    folder = Path(laser_folder)
    files  = sorted(p for p in folder.iterdir() if p.is_file())
    N      = len(files)
    print(f"  Found {N} laser images in {folder}")
    if N == 0:
        print("  ❌ No images — skipping.")
        return None

    # ── 2. Target resolution ──────────────────────────────────────────────────
    shapes = [tiff.imread(f).shape for f in files]
    min_h  = min(s[0] for s in shapes)
    min_w  = min(s[1] for s in shapes)
    if len({tuple(s) for s in shapes}) > 1:
        print(f"  ⚠️  Mixed resolutions — downscaling to ({min_h}×{min_w})")

    # ── 3. Dark frame (shared) ────────────────────────────────────────────────
    bg = get_master_dark(c.dark_folder, target_shape=(min_h, min_w))

    # ── 4. Load & upload to GPU ────────────────────────────────────────────────
    images_cpu = load_images_parallel(files, min_h, min_w, bg)
    images_gpu = cp.asarray(images_cpu)
    print(f"  ✅ Uploaded to {'GPU VRAM' if GPU_AVAILABLE else 'CPU RAM'}.")

    # ── 5. Main correlation curve ──────────────────────────────────────────────
    print(f"  Computing g₂(τ) for max_tau={c.max_tau}…")
    g  = to_numpy(compute_g_tau(images_gpu, c.max_tau))
    g -= g[-10:].mean()
    x  = np.arange(c.max_tau) / c.fps

    try:
        ibegin, iend = find_valid_fit_range(g, ibegin=0)
        print(f"  Dynamic fit window : [{ibegin}, {iend})  "
              f"({iend - ibegin} pts,  τ_max = {x[iend-1]:.4f} s)")
        slope, intercept, tau_c = fit_log_slope(g, x, ibegin, iend)
    except ValueError as e:
        print(f"  ❌ Fit failed: {e}")
        return None

    y_reg = slope * x + intercept
    print(f"  Slope  : {slope:.4f} 1/s")
    print(f"  τ_c    : {tau_c:.4f} s")

    # ── 6. Robustness sweep ────────────────────────────────────────────────────
    print("  Robustness sweep…")
    g_full   = to_numpy(compute_g_tau(images_gpu, max(c.max_frames_list)))
    tau_list = []

    for max_frame, bl_len in zip(c.max_frames_list, c.baseline_values):
        g1  = g_full[:max_frame].copy()
        g1 -= g1[-bl_len:].mean()
        x1  = np.arange(max_frame) / c.fps
        try:
            ib, ie = find_valid_fit_range(g1, ibegin=10)
            _, _, tc = fit_log_slope(g1, x1, ib, ie)
            tau_list.append(tc)
        except ValueError as e:
            print(f"    Skipping max_frame={max_frame}: {e}")

    # ── 7. Physics ────────────────────────────────────────────────────────────
    phys = compute_physics(T, n, nabla, c)
    
    # NEW: Calculate the real experimental D from your actual tau_c data
    D_exp = 1.0 / (tau_c * (phys['q'] ** 2))

    if tau_list:
        mean_tau = float(np.mean(tau_list))
        std_tau  = float(np.std(tau_list))
        print(f"  Mean τ_c : {mean_tau:.4f} s  ±  {std_tau:.4f} s  (robustness)")

    print(f"  q = {phys['q']:.4e} m⁻¹  |  D_theo = {phys['D_theo']:.4e} m²/s  |  D_exp = {D_exp:.4e} m²/s")

    # ── 8. c(t, τ) curves ─────────────────────────────────────────────────────
    tau_c_frames   = max(1, int(round(tau_c * c.fps)))
    c_t_tau_curves = {}
    for lag in sorted({1, 5, 10, 20, tau_c_frames}):
        if lag < N:
            c_t_tau_curves[lag] = to_numpy(compute_c_t_tau(images_gpu, lag))

    # ── 9. Per-temperature figure ──────────────────────────────────────────────
    out_dir = Path(c.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        rf"$T = {T:.2f}$ K  ({T_C:.1f} $^\circ$C)  —  "
        rf"$n = {n:.4f}$,  $\eta = {nabla:.3e}$ Pa$\cdot$s",
        fontsize=11,
    )

    # g₂(τ)−1
    axs[0].plot(x[:min(41, c.max_tau)], g[:min(41, c.max_tau)],
                "-o", label=r"$g_2(\tau)-1$")
    axs[0].axhline(0, color="k", ls="--", alpha=0.5)
    axs[0].set_xlabel(r"$\tau$ (s)")
    axs[0].set_ylabel(r"$g_2(\tau)-1$")
    axs[0].set_title("Intensity Correlation Function")
    axs[0].legend()

    # log fit
    axs[1].plot(x[ibegin:iend], np.log(g[ibegin:iend]),
                "-", label=r"$\log(g_2-1)$")
    axs[1].plot(x[ibegin:iend], y_reg[ibegin:iend],
                "--", color="r", label=rf"Fit  (slope = {slope:.2f})")
    axs[1].axvline(x[ibegin],   color="gray", ls=":", alpha=0.5)
    axs[1].axvline(x[iend - 1], color="gray", ls=":", alpha=0.5)
    axs[1].set_xlabel(r"$\tau$ (s)")
    axs[1].set_ylabel(r"$\log(g_2(\tau)-1)$")
    axs[1].set_title(f"Log fit  [{ibegin}, {iend})")
    axs[1].legend()

    # c(t, τ)
    cmap   = plt.get_cmap("plasma")
    n_lags = len(c_t_tau_curves)
    for idx, (lag, ct) in enumerate(sorted(c_t_tau_curves.items())):
        t_ax  = np.arange(len(ct)) / c.fps
        label = (
            rf"$\tau = {lag/c.fps:.3f}$ s"
            + (r"  ($\tau_c$)" if lag == tau_c_frames else "")
        )
        axs[2].plot(t_ax, ct, lw=0.8,
                    color=cmap(idx / max(n_lags - 1, 1)), label=label)
    axs[2].axhline(0, color="k", ls="--", alpha=0.4)
    axs[2].set_xlabel(r"Absolute time $t$ (s)")
    axs[2].set_ylabel(r"$c(t,\tau)$")
    axs[2].set_title(r"Instantaneous Correlation $c(t,\tau)$")
    axs[2].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / f"speckle_{T_C:.0f}C.png", dpi=150)
    plt.close(fig)   # free memory, avoid tkinter GC issues
    print(f"  ✅ Figure saved → speckle_{T_C:.0f}C.png")

    return {
        "T":     T,
        "T_C":   T_C,
        "n":     n,
        "nabla": nabla,
        "tau_c": tau_c,
        "q":     phys['q'],
        "D_theo": phys['D_theo'],
        "D_exp": D_exp,           # <--- Passed to the final plot!
        "a":     phys['a'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main — loop over all temperatures
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    c       = cfg
    results = []

    for laser_folder, T in c.experiments:
        result = run_experiment(laser_folder, T, c)
        if result is not None:
            results.append(result)

    if not results:
        print("\n⚠️  No valid results collected.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print(f"  {'T (°C)':>7}  {'T (K)':>7}  {'n':>8}  {'η (Pa·s)':>11}  "
          f"{'τ_c (s)':>10}  {'q (m⁻¹)':>12}  {'D_theo (m²/s)':>14}  {'D_exp (m²/s)':>14}")
    print("=" * 105)
    for r in results:
        print(
            f"  {r['T_C']:7.1f}  {r['T']:7.2f}  {r['n']:8.5f}  {r['nabla']:11.4e}  "
            f"  {r['tau_c']:10.4f}  {r['q']:12.4e}  {r['D_theo']:14.4e}  {r['D_exp']:14.4e}"
        )
    print("=" * 105)

    # ── D vs T plot ────────────────────────────────────────────────────────────
    if len(results) > 1:
        out_dir = Path(c.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        T_axis     = [r["T_C"]    for r in results]
        D_theo_ax  = [r["D_theo"] for r in results]
        D_exp_ax   = [r["D_exp"]  for r in results]
        nabla_axis = [r["nabla"]  for r in results]

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # D vs T (Now plots BOTH curves!)
        axs[0].plot(T_axis, D_theo_ax, "-o", color="steelblue", label="Theoretical D (Stokes-Einstein)")
        axs[0].plot(T_axis, D_exp_ax, "--s", color="darkorange", label="Experimental D (Measured)")
        
        for r in results:
            axs[0].annotate(
                f"{r['T_C']:.0f}°C",
                (r["T_C"], r["D_exp"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
            )
        axs[0].set_xlabel(r"Temperature $T$ ($^\circ$C)")
        axs[0].set_ylabel(r"$D$ (m$^2$/s)")
        axs[0].set_title(r"Diffusion coefficient $D$ vs $T$")
        axs[0].legend()

        # viscosity vs T (sanity check)
        axs[1].plot(T_axis, nabla_axis, "-s", color="tomato")
        axs[1].set_xlabel(r"Temperature $T$ ($^\circ$C)")
        axs[1].set_ylabel(r"$\eta$ (Pa$\cdot$s)")
        axs[1].set_title(r"Dynamic viscosity $\eta$ vs $T$")

        plt.tight_layout()
        plt.savefig(out_dir / "D_vs_T.png", dpi=150)
        plt.close(fig)
        print(f"\n  ✅ Summary figure saved → D_vs_T.png")


if __name__ == "__main__":
    main()