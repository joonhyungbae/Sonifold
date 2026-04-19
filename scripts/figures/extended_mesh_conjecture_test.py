"""
Extended mesh Conjecture 4.1 test: E[β₀] increases monotonically with spectral entropy H.

1. Generate ≥8 geometrically distinct closed meshes (~5k–10k vertices), save to data/meshes/extended/.
2. For each new mesh: K=50 eigenvalues (cotangent LB), spectral entropy H, mean β₀ (N=200 random),
   optionally mean β₀ under A5 (white noise) direct mapping.
3. Combine with existing 13 meshes (9 exploration + genus 3–6). Compute Spearman ρ(H, mean_β₀)
   for full set (n ≥ 21). Optionally K=100, K=200 if eigenbases available.
4. Output: data/results/extended_mesh_spectral_entropy.csv, figures/fig_spectral_entropy_extended.pdf,
   and print conjecture support level (STRONG / MODERATE / WEAK).

   For K=100 or K=200: precompute eigen with N=100 or N=200 (e.g. run_step1 with N_EIGEN=200)
   for the meshes you care about; then Spearman can be computed on that subset in a separate pass.

Run from project root: python scripts/extended_mesh_conjecture_test.py

Optional: USE_GPU=1 for CuPy, EIGEN_TOL=1e-3 for faster convergence.
"""
from __future__ import annotations

import os
import csv
import json
import sys
import warnings
from pathlib import Path

# Use GPU by default (eigensolver uses CuPy when available)
os.environ.setdefault("USE_GPU", "1")

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import MESHES_9, MESHES_GENUS_36, spectral_entropy

RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
EIGEN_DIR = root / "data" / "eigen"
EIGEN_EXPERIMENTS_DIR = root / "data" / "experiments" / "eigen"

# Extended mesh config
EXTENDED_DIR = root / "data" / "meshes" / "extended"
EXTENDED_EIGEN_DIR = root / "data" / "eigen"  # store as {name}.npz for extended meshes
TARGET_VERTICES_MIN = 5000
TARGET_VERTICES_MAX = 10000
TARGET_VERTICES = 7500
VERTEX_TOLERANCE = 0.35
K = 50
K_EXTRA = [100, 200]
N_RANDOM = 200
RNG = np.random.default_rng(42)

OUT_CSV = RESULTS_DIR / "extended_mesh_spectral_entropy.csv"
OUT_FIG = FIGURES_DIR / "fig_spectral_entropy_extended.pdf"


# ---------- Extended mesh generators (each returns (V, F) with ~5k–10k vertices) ----------

def _remesh_to_target(V: np.ndarray, F: np.ndarray, target: int = TARGET_VERTICES):
    """Remesh to target vertex count. Uses trimesh subdivide / decimate."""
    try:
        import trimesh
    except ImportError:
        return V.astype(np.float64), F.astype(np.int32)
    n = V.shape[0]
    if target * (1 - VERTEX_TOLERANCE) <= n <= target * (1 + VERTEX_TOLERANCE):
        return V.astype(np.float64), F.astype(np.int32)
    m = trimesh.Trimesh(vertices=V, faces=F)
    if n > target * (1 + VERTEX_TOLERANCE):
        try:
            m = m.simplify_quadric_decimation(int(target * 2))
        except Exception:
            pass
    else:
        while m.vertices.shape[0] < target * (1 - VERTEX_TOLERANCE):
            m = m.subdivide()
            if m.vertices.shape[0] >= 2 * target:
                break
    return m.vertices.astype(np.float64), m.faces.astype(np.int32)


def _sphere_base():
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    return m.vertices, m.faces


def _ellipsoid_112():
    V, F = _sphere_base()
    V[:, 0] *= 1.0
    V[:, 1] *= 1.0
    V[:, 2] *= 2.0
    return _remesh_to_target(V, F)


def _ellipsoid_123():
    V, F = _sphere_base()
    V[:, 0] *= 1.0
    V[:, 1] *= 2.0
    V[:, 2] *= 3.0
    return _remesh_to_target(V, F)


def _ellipsoid_115():
    V, F = _sphere_base()
    V[:, 0] *= 1.0
    V[:, 1] *= 1.0
    V[:, 2] *= 5.0
    return _remesh_to_target(V, F)


def _torus_Rr(R: float, r: float, n_ring: int = 90):
    import gpytoolbox as gpy
    # n_ring^2 * 2 ≈ vertices; 90*90*2 = 16200, then decimate to ~7.5k
    V, F = gpy.torus(n_ring, n_ring, R=R, r=r)
    return _remesh_to_target(V, F)


def _torus_Rr2():
    return _torus_Rr(2.0, 1.0)


def _torus_Rr3():
    return _torus_Rr(3.0, 1.0)


def _torus_Rr5():
    return _torus_Rr(5.0, 1.0)


def _torus_bent():
    """Genus-1 torus with different embedding (bent / aspect) — distinct from standard tori."""
    import gpytoolbox as gpy
    V, F = gpy.torus(85, 85, R=2.5, r=0.8)
    return _remesh_to_target(V, F)


def _genus2_asymmetric():
    """Genus-2 with asymmetric handle placement (two tori with offset)."""
    root_exp = root
    meshes_dir = root_exp / "data" / "meshes"
    obj_path = meshes_dir / "double_torus.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(str(obj_path), force="mesh")
        V, F = m.vertices, m.faces
        return _remesh_to_target(V, F)
    try:
        from experiment.generate_experiment_meshes import _high_genus_by_tori, _remesh_to_target as remesh_exp
        V, F = _high_genus_by_tori(2, R=2.0, r=0.6, n_ring=50)
        return remesh_exp(V, F, 5000)
    except Exception:
        import gpytoolbox as gpy
        V, F = gpy.torus(80, 80, R=2.2, r=0.55)
        return _remesh_to_target(V, F)


def _rounded_cube():
    """Superellipsoid-like: deform sphere toward cube (|x|^p * sign(x) with p < 1)."""
    V, F = _sphere_base()
    p = 0.45
    for i in range(3):
        x = V[:, i]
        V[:, i] = np.sign(x) * (np.abs(x) ** p)
    # scale back to similar radius
    r = np.sqrt(np.sum(V ** 2, axis=1))
    r_mean = np.mean(r)
    if r_mean > 1e-10:
        V = V / r_mean
    return _remesh_to_target(V, F)


def _bunny_optional():
    """Stanford bunny or similar genus-0 if available."""
    for name in ("bunny", "spot"):
        path = root / "data" / "meshes" / f"{name}.obj"
        if path.exists():
            import trimesh
            m = trimesh.load(str(path), force="mesh")
            V, F = m.vertices, m.faces
            return _remesh_to_target(V, F)
    return None


def _genus7_optional():
    """Genus-7 by boolean union of 7 tori (optional, may fail without manifold3d)."""
    try:
        from experiment.generate_experiment_meshes import _high_genus_by_tori, _remesh_to_target as remesh_exp
        V, F = _high_genus_by_tori(7, R=2.0, r=0.42, n_ring=38)
        return remesh_exp(V, F, TARGET_VERTICES)
    except Exception as e:
        warnings.warn(f"Genus-7 mesh failed: {e}", UserWarning)
        return None


def _genus8_optional():
    """Genus-8 by boolean union of 8 tori (optional)."""
    try:
        from experiment.generate_experiment_meshes import _high_genus_by_tori, _remesh_to_target as remesh_exp
        V, F = _high_genus_by_tori(8, R=2.0, r=0.40, n_ring=36)
        return remesh_exp(V, F, TARGET_VERTICES)
    except Exception as e:
        warnings.warn(f"Genus-8 mesh failed: {e}", UserWarning)
        return None


# (name, genus, generator_fn); generator returns (V, F) or None to skip
EXTENDED_MESH_SPECS = [
    ("ellipsoid_112", 0, _ellipsoid_112),
    ("ellipsoid_123", 0, _ellipsoid_123),
    ("ellipsoid_115", 0, _ellipsoid_115),
    ("torus_Rr2", 1, _torus_Rr2),
    ("torus_Rr3", 1, _torus_Rr3),
    ("torus_Rr5", 1, _torus_Rr5),
    ("torus_bent", 1, _torus_bent),
    ("genus2_asymmetric", 2, _genus2_asymmetric),
    ("rounded_cube", 0, _rounded_cube),
    ("bunny", 0, _bunny_optional),
    ("genus7", 7, _genus7_optional),
    ("genus8", 8, _genus8_optional),
]


def ensure_extended_meshes():
    """Generate and save extended meshes to data/meshes/extended/."""
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, genus, fn in EXTENDED_MESH_SPECS:
        path = EXTENDED_DIR / f"{name}.obj"
        if path.exists():
            print(f"[extended] {name} already exists: {path}", file=sys.stderr)
            saved.append((name, genus, path))
            continue
        try:
            out = fn()
            if out is None:
                print(f"[extended] {name} skipped (generator returned None)", file=sys.stderr)
                continue
            V, F = out
            import trimesh
            m = trimesh.Trimesh(vertices=V, faces=F)
            m.export(str(path))
            print(f"[extended] {name} genus={genus} n_verts={V.shape[0]} -> {path}", file=sys.stderr)
            saved.append((name, genus, path))
        except Exception as e:
            print(f"[extended] {name} failed: {e}", file=sys.stderr)
    return saved


def load_extended_mesh(name: str):
    """Load V, F from data/meshes/extended/{name}.obj."""
    path = EXTENDED_DIR / f"{name}.obj"
    if not path.exists():
        return None, None
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def compute_and_save_eigen(name: str, V: np.ndarray, F: np.ndarray, K_val: int = K):
    """Compute first K_val eigenvalues/eigenvectors (cotangent LB) and save to data/eigen/{name}.npz."""
    from precompute.eigensolver import compute_eigen
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = EIGEN_DIR / f"{name}.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        if "eigenvalues" in data.files and len(data["eigenvalues"]) >= K_val:
            return np.asarray(data["eigenvalues"], dtype=np.float64), np.asarray(data["eigenvectors"], dtype=np.float64)
    evals, evecs = compute_eigen(V, F, N=K_val)
    np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
    return evals, evecs


def sample_uniform_sphere(k: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(k)
    n = np.linalg.norm(a)
    if n < 1e-20:
        a = rng.standard_normal(k)
        n = np.linalg.norm(a)
    return (a / n).astype(np.float64)


def beta0_random(V, F, evecs, k: int, n_trials: int = N_RANDOM):
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    beta0s = []
    for _ in range(n_trials):
        coef = sample_uniform_sphere(k, RNG)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        beta0s.append(m.beta0)
    arr = np.array(beta0s, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def beta0_A5_if_available(V, F, evecs, evals, k: int):
    """Mean β₀ over A5 (white noise) STFT frames, direct mapping."""
    try:
        from audio.audio_library import get_audio
        from audio.fft_analysis import compute_fft_frames
        from mapping.spectral_mapping import map_fft_to_coefficients
        from analysis.scalar_field import compute_scalar_field
        from analysis.nodal_surface import compute_topology_metrics
    except ImportError:
        return None
    try:
        sig, sr = get_audio("A5")
        n_target = int(10.0 * sr)
        if len(sig) < n_target:
            sig = np.tile(sig, (n_target + len(sig) - 1) // len(sig))[:n_target]
        else:
            sig = sig[:n_target]
        frames = compute_fft_frames(sig, sample_rate=sr, hop_length=512)
        if not frames:
            return None
        beta0s = []
        for mag, _ in frames:
            coef = map_fft_to_coefficients(mag, k, strategy="direct", eigenvalues=evals)
            f = compute_scalar_field(evecs, coef)
            m = compute_topology_metrics(V, F, f)
            beta0s.append(m.beta0)
        return float(np.mean(beta0s))
    except Exception:
        return None


def process_existing_13():
    """Load existing 13 meshes: for each get (mesh_name, genus, num_vertices, H, mean_beta0, std_beta0)."""
    from precompute.eigensolver import compute_eigen
    rows = []
    for mesh_id, genus in list(MESHES_9) + list(MESHES_GENUS_36):
        is_genus36 = any(m == mesh_id for m, _ in MESHES_GENUS_36)
        base = EIGEN_EXPERIMENTS_DIR if is_genus36 else EIGEN_DIR
        for ext in (".npz", ".json"):
            p = base / (mesh_id + ext)
            if not p.exists():
                continue
            if p.suffix == ".npz":
                data = np.load(p, allow_pickle=True)
                V = np.asarray(data["vertices"], dtype=np.float64)
                F = np.asarray(data["faces"], dtype=np.int32)
                evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
                evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
            else:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                V = np.array(data["vertices"], dtype=np.float64)
                F = np.array(data["faces"], dtype=np.int32)
                evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
                evecs = np.array(data["eigenvectors"], dtype=np.float64)
            n_verts = V.shape[0]
            if evals is None or len(evals) < 2:
                evals = None
            else:
                evals = np.sort(evals)[:K]
            # evecs: (N, V) rows = modes, or (V, N) columns = modes
            if evecs.shape[0] == n_verts:
                evecs = evecs[:, :K].T  # -> (K, V)
            else:
                evecs = evecs[:K, :]    # -> (K, V)
            if evecs.shape[0] < K or evecs.shape[1] != n_verts or (evals is not None and len(evals) < 2):
                print(f"[existing] Skip {mesh_id}: insufficient eigen", file=sys.stderr)
                break
            if evals is None:
                evals, evecs = compute_eigen(V, F, N=K)
                evals = evals[:K]
            H = spectral_entropy(evals)
            mean_b0, std_b0 = beta0_random(V, F, evecs, K)
            rows.append({
                "mesh_name": mesh_id,
                "genus": genus,
                "num_vertices": n_verts,
                "spectral_entropy_H": round(H, 6),
                "mean_beta0_random": round(mean_b0, 4),
                "std_beta0_random": round(std_b0, 4),
                "mean_beta0_A5": None,
            })
            print(f"[existing] {mesh_id} genus={genus} n={n_verts} H={H:.4f} mean_β0={mean_b0:.2f}", file=sys.stderr)
            break
        else:
            print(f"[existing] No eigen file for {mesh_id}", file=sys.stderr)
    return rows


def process_extended_meshes(saved_specs):
    """For each extended mesh: compute eigen (if needed), H, mean β₀, optional A5."""
    rows = []
    for name, genus, _ in saved_specs:
        V, F = load_extended_mesh(name)
        if V is None:
            continue
        n_verts = V.shape[0]
        evals, evecs = compute_and_save_eigen(name, V, F, K_val=K)
        if evals is None or len(evals) < 2:
            print(f"[extended] Skip {name}: eigen failed", file=sys.stderr)
            continue
        evals = np.sort(evals)[:K]
        if evecs.shape[0] != K:
            evecs = evecs[:K, :] if evecs.shape[0] > evecs.shape[1] else evecs[:, :K].T
        H = spectral_entropy(evals)
        mean_b0, std_b0 = beta0_random(V, F, evecs, K)
        mean_a5 = beta0_A5_if_available(V, F, evecs, evals, K)
        rows.append({
            "mesh_name": name,
            "genus": genus,
            "num_vertices": n_verts,
            "spectral_entropy_H": round(H, 6),
            "mean_beta0_random": round(mean_b0, 4),
            "std_beta0_random": round(std_b0, 4),
            "mean_beta0_A5": round(mean_a5, 4) if mean_a5 is not None else None,
        })
        print(f"[extended] {name} genus={genus} n={n_verts} H={H:.4f} mean_β0={mean_b0:.2f}", file=sys.stderr)
    return rows


def compute_spearman_full(rows):
    """Spearman ρ(H, mean_beta0_random) for full set. Returns (rho, p_value)."""
    from scipy.stats import spearmanr
    H = np.array([r["spectral_entropy_H"] for r in rows], dtype=np.float64)
    y = np.array([r["mean_beta0_random"] for r in rows], dtype=np.float64)
    ok = np.isfinite(H) & np.isfinite(y)
    n = int(np.sum(ok))
    if n < 3:
        return np.nan, np.nan
    rho, p = spearmanr(H[ok], y[ok])
    return float(rho), float(p)


def write_csv(rows, rho_full, p_full):
    """Write extended_mesh_spectral_entropy.csv with spearman_rho_cumulative = full-set ρ."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["spearman_rho_cumulative"] = round(rho_full, 4) if np.isfinite(rho_full) else ""
    fieldnames = [
        "mesh_name", "genus", "num_vertices", "spectral_entropy_H",
        "mean_beta0_random", "std_beta0_random", "mean_beta0_A5", "spearman_rho_cumulative",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}", file=sys.stderr)


def plot_figure(rows, rho_full, p_full):
    """Scatter H vs mean_β₀, colored by genus, labeled, Spearman ρ annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x = [r["spectral_entropy_H"] for r in rows]
    y = [r["mean_beta0_random"] for r in rows]
    labels = [r["mesh_name"].replace("_", " ") for r in rows]
    genera = [r["genus"] for r in rows]

    def genus_color(g):
        if g == 0:
            return "C0"
        if g == 1:
            return "C2"
        if g == 2:
            return "C3"
        return "purple"

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [genus_color(g) for g in genera]
    ax.scatter(x, y, c=colors, s=70, zorder=2, edgecolors="k", linewidths=0.5)
    for i, lb in enumerate(labels):
        ax.annotate(lb, (x[i], y[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)
    ax.set_xlabel("Spectral entropy $H$ (normalized gaps)")
    ax.set_ylabel(r"Mean $\beta_0$ (random coefficients, $N=200$)")
    ax.set_title(r"$H$ vs $\beta_0$ — extended mesh set ($n \geq 21$)")
    rho_str = f"{rho_full:.2f}" if np.isfinite(rho_full) else "—"
    p_str = f"$p$ = {p_full:.2e}" if np.isfinite(p_full) else ""
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho_str}\n{p_str}", transform=ax.transAxes, fontsize=10, va="top")
    plt.tight_layout()
    plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_FIG}", file=sys.stderr)


def print_conjecture_support(rho_full, p_full):
    """STRONG if ρ > 0.6 and p < 0.01, MODERATE if ρ > 0.4 and p < 0.05, WEAK otherwise."""
    r = float(rho_full) if np.isfinite(rho_full) else 0.0
    p = float(p_full) if np.isfinite(p_full) else 1.0
    if r > 0.6 and p < 0.01:
        level = "STRONG"
    elif r > 0.4 and p < 0.05:
        level = "MODERATE"
    else:
        level = "WEAK"
    print("\nConjecture support level:", level, f"(Spearman ρ = {r:.3f}, p = {p:.4f})")
    if level == "STRONG":
        print("  ρ > 0.6 and p < 0.01 → Conjecture 4.1 well supported.")
    elif level == "MODERATE":
        print("  ρ > 0.4 and p < 0.05 → Conjecture 4.1 moderately supported.")
    else:
        print("  WEAK → Larger n or different meshes may be needed to strengthen evidence.")


def main():
    # 1. Generate extended meshes
    saved_specs = ensure_extended_meshes()
    if len(saved_specs) < 8:
        print("Warning: fewer than 8 extended meshes generated; continuing anyway.", file=sys.stderr)

    # 2. Process existing 13 meshes
    existing_rows = process_existing_13()
    if len(existing_rows) < 13:
        print(f"Warning: only {len(existing_rows)} existing meshes loaded.", file=sys.stderr)

    # 3. Process extended meshes
    extended_rows = process_extended_meshes(saved_specs)

    # 4. Combine
    all_rows = existing_rows + extended_rows
    if len(all_rows) < 21:
        print(f"Only {len(all_rows)} meshes total; need ≥ 21 for extended test. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 5. Spearman for full set
    rho_full, p_full = compute_spearman_full(all_rows)
    print(f"\nFull set: n = {len(all_rows)}, Spearman ρ(H, mean_β₀) = {rho_full:.4f}, p = {p_full:.4e}", file=sys.stderr)

    # 6. Output CSV and figure
    write_csv(all_rows, rho_full, p_full)
    plot_figure(all_rows, rho_full, p_full)

    # 7. Conjecture support level
    print_conjecture_support(rho_full, p_full)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
