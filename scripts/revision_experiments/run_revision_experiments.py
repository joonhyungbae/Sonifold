"""
Revision round 1 experiments for Sonifold (JMathArts).

Runs four experiments in response to Reviewer 2:
  E1  Mesh rescaling invariance   -- rescale sphere/torus by α ∈ {0.5,1,2,4}
                                     and verify that β0/A_ratio/S are identical
                                     (expected: exactly equal, because the
                                     mapping is ordinal and never reads λ).
  E2  Pitch transposition         -- pitch-shift piano (A3) by
                                     {-12,-5,-1,0,+1,+5,+12} semitones and
                                     measure β0 on sphere, torus, double torus.
  E3  Frequency-matched mapping   -- compare 'direct' (ordinal) vs 'freqmatch'
                                     (Hz ↔ √λ) on four audio stimuli × three
                                     meshes, logging β0/A_ratio/S distributions.
  E4  Same-genus / different metric -- genus-1 realised as three embeddings
                                     (R=3,r=1), (R=2,r=0.6), (R=1.5,r=1.0),
                                     same audio → quantify how β0 varies
                                     across geometric realisations of genus 1.

Outputs:
  data/experiments/revision/E1_rescale.json
  data/experiments/revision/E2_transpose.json
  data/experiments/revision/E3_freqmatch.json
  data/experiments/revision/E4_sameg.json

These are consumed by scripts/figures/generate_revision_figures.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on sys.path when run standalone
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force CPU unless caller overrides (GPU path requires CuPy)
os.environ.setdefault("USE_GPU", "0")
# Tight ARPACK tolerance for revision experiments: round 1 used 1e-3 which
# made E1 results dominated by floating-point drift in the cotangent assembly
# at large rescale factors; 1e-6 allows the symmetric-form scaling argument
# A_α = α^{-2} A_1 to be tested at the level it actually predicts.
os.environ.setdefault("EIGEN_TOL", "1e-6")

from analysis.nodal_surface import compute_topology_metrics
from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from precompute.eigensolver import compute_eigen
from precompute.mesh_library import get_mesh
try:
    import gpytoolbox as gpy
except ImportError:
    gpy = None

try:
    import trimesh
except ImportError:
    trimesh = None

N_MODES = 50
EPS_RATIO = 0.05
OUT_DIR = _ROOT / "data" / "experiments" / "revision"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics_of(V, F, coefs, eigvecs):
    """Compute β0, A_ratio for a given coefficient vector."""
    field = eigvecs.T @ coefs  # (V,)
    m = compute_topology_metrics(V, F, field, threshold_ratio=EPS_RATIO)
    return {"beta0": int(m.beta0), "A_ratio": float(m.A_ratio)}


def compute_mesh_and_eigen(name_or_params, use_cache: bool = True):
    """Returns (V, F, eigenvalues, eigenvectors). name_or_params either str or dict.

    For plain string names and use_cache=True, loads from data/eigen/<name>.npz
    if present (25k-vert precomputed bases used by the web application).
    For scaled/parametric meshes, always computes fresh.
    """
    cache_dir = _ROOT / "data" / "eigen"
    if isinstance(name_or_params, str):
        if use_cache:
            cache = cache_dir / f"{name_or_params}.npz"
            if cache.exists():
                d = np.load(cache)
                V = d["vertices"].astype(np.float64)
                F = d["faces"].astype(np.int32)
                lam = d["eigenvalues"].astype(np.float64)[:N_MODES]
                vec = d["eigenvectors"].astype(np.float64)[:N_MODES, :]
                return V, F, lam, vec
        V, F = get_mesh(name_or_params)
    elif isinstance(name_or_params, dict) and name_or_params["type"] == "torus":
        if gpy is None:
            raise ImportError("gpytoolbox required for torus")
        R, r = name_or_params["R"], name_or_params["r"]
        nR = name_or_params.get("nR", 100)
        nr = name_or_params.get("nr", 100)
        V, F = gpy.torus(nR, nr, R=R, r=r)
        V = V.astype(np.float64)
        F = F.astype(np.int32)
    elif isinstance(name_or_params, dict) and name_or_params["type"] == "scaled":
        V, F = get_mesh(name_or_params["base"])
        V = V * float(name_or_params["alpha"])
    else:
        raise ValueError(f"Unknown mesh spec {name_or_params}")
    eigvals, eigvecs = compute_eigen(V, F, N=N_MODES)
    return V, F, eigvals, eigvecs


def audio_to_frame_coefs(signal, eigvals, strategy="direct", hop=512):
    """Return list of coefficient vectors, one per STFT frame."""
    frames = compute_fft_frames(signal, sample_rate=44100, hop_length=hop)
    coefs = []
    for mag, _ in frames:
        c = map_fft_to_coefficients(
            mag, N_MODES, strategy=strategy, eigenvalues=eigvals
        )
        coefs.append(c)
    return coefs


def mean_metrics_over_frames(V, F, coef_list, eigvecs):
    beta0s, arats = [], []
    for c in coef_list:
        field = eigvecs.T @ c
        m = compute_topology_metrics(V, F, field, threshold_ratio=EPS_RATIO)
        beta0s.append(m.beta0)
        arats.append(m.A_ratio)
    return {
        "beta0_mean": float(np.mean(beta0s)),
        "beta0_std": float(np.std(beta0s)),
        "A_ratio_mean": float(np.mean(arats)),
        "A_ratio_std": float(np.std(arats)),
        "n_frames": len(coef_list),
    }


# --------------------------------------------------------------------------
# E1  Mesh rescaling invariance
# --------------------------------------------------------------------------
def run_E1():
    log("E1: mesh rescaling invariance")
    results = {"description": "rescale sphere/torus by α, compare β0/A_ratio across α",
               "stimuli": ["A1", "A3", "A5"], "alphas": [0.5, 1.0, 2.0, 4.0],
               "shapes": ["sphere", "torus"], "records": []}
    for shape in results["shapes"]:
        for alpha in results["alphas"]:
            log(f"  E1 {shape} α={alpha}")
            V, F, lam, U = compute_mesh_and_eigen({"type": "scaled", "base": shape, "alpha": alpha})
            for aid in results["stimuli"]:
                sig, sr = get_audio(aid)
                coefs = audio_to_frame_coefs(sig, lam, strategy="direct")
                r = mean_metrics_over_frames(V, F, coefs, U)
                r.update({"shape": shape, "alpha": alpha, "audio": aid,
                          "lambda_max": float(lam.max())})
                results["records"].append(r)
    out = OUT_DIR / "E1_rescale.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"  E1 wrote {out}")
    return results


# --------------------------------------------------------------------------
# E2  Pitch transposition
# --------------------------------------------------------------------------
def run_E2():
    log("E2: pitch transposition")
    import librosa
    results = {"description": "pitch-shift A3 piano by s semitones, measure β0 drift",
               "semitones": [-12, -5, -1, 0, 1, 5, 12],
               "shapes": ["sphere", "torus", "double_torus"],
               "records": []}
    sig_orig, sr = get_audio("A3")
    log(f"  A3 loaded, {len(sig_orig)} samples @ {sr} Hz")
    # Pre-transpose the audio (same for all meshes)
    shifted = {}
    for s in results["semitones"]:
        if s == 0:
            shifted[s] = sig_orig.copy()
        else:
            log(f"  shifting {s:+d} semitones")
            shifted[s] = librosa.effects.pitch_shift(
                y=sig_orig.astype(np.float32), sr=sr, n_steps=s
            ).astype(np.float64)
    for shape in results["shapes"]:
        log(f"  E2 mesh={shape}")
        V, F, lam, U = compute_mesh_and_eigen(shape)
        for s in results["semitones"]:
            coefs = audio_to_frame_coefs(shifted[s], lam, strategy="direct")
            r = mean_metrics_over_frames(V, F, coefs, U)
            r.update({"shape": shape, "semitones": s})
            results["records"].append(r)
    out = OUT_DIR / "E2_transpose.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"  E2 wrote {out}")
    return results


# --------------------------------------------------------------------------
# E3  Frequency-matched mapping ablation
# --------------------------------------------------------------------------
def run_E3():
    log("E3: freq-matched ablation")
    results = {"description": "direct vs freqmatch on same audio",
               "strategies": ["direct", "freqmatch"],
               "stimuli": ["A1", "A3", "A5", "A7"],
               "shapes": ["sphere", "torus", "double_torus"],
               "records": []}
    for shape in results["shapes"]:
        log(f"  E3 mesh={shape}")
        V, F, lam, U = compute_mesh_and_eigen(shape)
        for aid in results["stimuli"]:
            sig, sr = get_audio(aid)
            for strat in results["strategies"]:
                coefs = audio_to_frame_coefs(sig, lam, strategy=strat)
                r = mean_metrics_over_frames(V, F, coefs, U)
                r.update({"shape": shape, "audio": aid, "strategy": strat})
                results["records"].append(r)
    out = OUT_DIR / "E3_freqmatch.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"  E3 wrote {out}")
    return results


# --------------------------------------------------------------------------
# E4  Same-genus / different metric (genus 1)
# --------------------------------------------------------------------------
def run_E4():
    log("E4: same-genus, different metric (genus-1)")
    specs = [
        {"label": "torus_3_1",   "type": "torus", "R": 3.0, "r": 1.0, "nR": 100, "nr": 100},
        {"label": "torus_2_0.6", "type": "torus", "R": 2.0, "r": 0.6, "nR": 100, "nr": 100},
        {"label": "torus_1.5_1", "type": "torus", "R": 1.5, "r": 1.0, "nR": 100, "nr": 100},
    ]
    results = {"description": "three genus-1 embeddings, compare β0 dispersion",
               "stimuli": ["A1", "A3", "A5"],
               "specs": [s["label"] for s in specs],
               "records": []}
    for spec in specs:
        log(f"  E4 mesh={spec['label']}")
        V, F, lam, U = compute_mesh_and_eigen(spec)
        for aid in results["stimuli"]:
            sig, sr = get_audio(aid)
            coefs = audio_to_frame_coefs(sig, lam, strategy="direct")
            r = mean_metrics_over_frames(V, F, coefs, U)
            r.update({"label": spec["label"], "audio": aid,
                      "lambda_min": float(lam.min()),
                      "lambda_max": float(lam.max()),
                      "R": spec["R"], "r": spec["r"]})
            results["records"].append(r)
    out = OUT_DIR / "E4_sameg.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"  E4 wrote {out}")
    return results


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["E1", "E2", "E3", "E4"]
    runners = {"E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4}
    for name in which:
        if name not in runners:
            print(f"unknown experiment {name}")
            continue
        t0 = time.time()
        runners[name]()
        log(f"{name} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
