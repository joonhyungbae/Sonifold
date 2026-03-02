"""
Generate paper figures and table data (only figures referenced in paper.tex).
Run from project root:  python scripts/figures/generate_figures.py

Output: fig2_eigenvalues.pdf, fig3_nodal_gallery.pdf, fig8_spherical_harmonics.pdf,
        fig_temporal_persistence.pdf, tab1_beta0_direct.csv, web.png (placeholder if capture not run).
Real web app screenshot:  cd webapp && npm run dev   then  python scripts/figures/capture_demo_screenshot.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
figures_dir = root / "figures"

def load_eigenvalues(mesh_name: str):
    npz_path = root / "data" / "eigen" / f"{mesh_name}.npz"
    json_path = root / "data" / "eigen" / f"{mesh_name}.json"
    if npz_path.exists():
        import numpy as np
        d = np.load(npz_path, allow_pickle=True)
        return list(d["eigenvalues"].flat[:50])
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get("eigenvalues", [])[:50]
    return None

def fig2_eigenvalues():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rc("font", family="serif")
    meshes = ["sphere", "torus", "cube", "ellipsoid", "double_torus", "flat_plate",
              "tetrahedron", "octahedron", "icosahedron"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(meshes)))
    markers = ["o", "s", "^", "v", "D", "p", "h", "*", "X"]
    fig, ax = plt.subplots(figsize=(7, 4))
    spectral_gaps = {}
    for i, name in enumerate(meshes):
        ev = load_eigenvalues(name)
        if ev is None:
            continue
        ev = np.array(ev)
        if len(ev) >= 2:
            spectral_gaps[name] = float(ev[1] / (ev[0] + 1e-20))
        k = np.arange(1, len(ev) + 1)
        ax.plot(k, ev, label=name.replace("_", " "), alpha=0.85, color=colors[i],
                marker=markers[i % len(markers)], markevery=max(1, len(ev) // 12),
                markersize=4, linewidth=1.2)
    ax.set_xlabel("Eigenvalue index $k$")
    ax.set_ylabel("$\\lambda_k$")
    ax.set_title("Eigenvalue distribution by mesh")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 50.5)
    # Annotate large spectral gap (e.g. torus or sphere)
    if "sphere" in spectral_gaps and spectral_gaps["sphere"] > 2.5:
        ev_s = np.array(load_eigenvalues("sphere"))
        if ev_s is not None and len(ev_s) >= 2:
            ax.annotate("large $\\gamma$", xy=(2, ev_s[1]), fontsize=7, color="gray")
    fig.tight_layout()
    out = root / "figures" / "fig2_eigenvalues.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out)

def tables():
    import pandas as pd
    csv_path = root / "data" / "results" / "results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    # Tab 1: subset (e.g. direct strategy, selected meshes/audio)
    sub = df[df["strategy"] == "direct"].pivot_table(
        index="mesh", columns="audio", values="beta0", aggfunc="first"
    )
    out1 = root / "figures" / "tab1_beta0_direct.csv"
    out1.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out1)
    print("Saved", out1)
    # Tab 2: spectral summary per mesh (placeholder - would need npz/json)
    print("Tab 2: add spectral_gap, weyl_density per mesh from hypothesis_test data")

def run_fig_script(name: str) -> bool:
    """Run a figure script in scripts/figures/; return True if success."""
    script = Path(__file__).resolve().parent / name
    if not script.exists():
        print("Skip (not found):", script, file=sys.stderr)
        return False
    r = subprocess.run([sys.executable, str(script)], cwd=str(root))
    return r.returncode == 0


def web_placeholder():
    """Ensure figures/web.png exists (paper \\includegraphics{figures/web.png}); create placeholder if missing or tiny."""
    web_p = figures_dir / "web.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if web_p.exists() and web_p.stat().st_size >= 500:
        print("web.png present (use capture_demo_screenshot.py for real screenshot)")
        return
    try:
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont
    except ImportError:
        print("Skip web.png placeholder: pip install Pillow", file=sys.stderr)
        return
    img = Image.new("RGB", (800, 500), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    text = "Web app screenshot\nRun: python figures/capture_demo_screenshot.py\n(webapp must be running)"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((40, 40), text, fill=(80, 80, 80), font=font)
    img.save(web_p)
    print("Saved placeholder", web_p, "- replace with real screenshot when webapp is running")


def main():
    fig2_eigenvalues()
    if run_fig_script("fig3_nodal_gallery.py"):
        print("fig3_nodal_gallery.pdf done")
    if run_fig_script("fig_platonic_gallery.py"):
        print("fig_platonic_gallery.pdf done")
    if run_fig_script("fig8_spherical_harmonics.py"):
        print("fig8_spherical_harmonics.pdf done")
    tables()
    if run_fig_script("fig_temporal_persistence.py"):
        print("fig_temporal_persistence.pdf done")
    web_placeholder()


if __name__ == "__main__":
    main()
