# Sonifold: Audio-Driven Nodal Surfaces on 3D Meshes

Audio-driven nodal patterns on 3D shapes via Laplace–Beltrami eigenfunctions; the shape acts as a geometry-dependent spectral filter. Companion to the paper *Sonifold: Audio-Driven Nodal Surfaces on 3D Meshes* (Journal of Mathematics and the Arts).

## Setup

Python 3.10; `pip install -r requirements.txt` or `conda env create -f environment.yml` then `conda activate sonifold`. GPU optional: use `cupy-cuda12x` (or `cupy-cuda11x`) for faster Step 1; otherwise run Step 1 with `USE_GPU=0`.

## Pipeline (run from project root)

| Step | Command |
|------|---------|
| 1 — LB eigenpairs → `data/eigen/` | `./step1.sh` or `USE_GPU=0 python -m precompute.run_step1` |
| 2–3 — Verify (optional) | `./step2.sh`, `./step3.sh` |
| 4 — Full experiment → `data/results/` | `./step4.sh` |
| 4b — Extended experiments + figures | `./step4b.sh` (see script for deps) |
| 5 — Web app (local) | `./step5.sh` |

**Paper figures:** after Step 1 and 4, run `python scripts/figures/generate_figures.py` (output in `figures/`).

**Data:** `data/eigen/` and `data/results/` are gitignored; create them with Step 1 and 4. Optional WAVs for A3/A4/A7: see `data/audio/README.md`. Full layout: [data/README.md](data/README.md).

## License

MIT ([LICENSE](LICENSE)). Paper/figures may have separate terms.

**Layout:** Core code: `precompute/`, `audio/`, `mapping/`, `analysis/`. Runs: `experiment/`. Scripts: `scripts/figures/`, `scripts/analysis/`. See [docs/STRUCTURE.md](docs/STRUCTURE.md).
