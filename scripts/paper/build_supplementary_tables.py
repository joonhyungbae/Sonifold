"""
Build the LaTeX table fragments used by sonifold_paper/supplementary.tex.

Every table in the supplementary PDF is emitted here from the CSV/JSON under
data/results/ and data/experiments/revision/, so that re-running the analysis
pipeline and re-running this script keeps the PDF in step with the data. No
number is transcribed by hand.

Run from project root:  python scripts/paper/build_supplementary_tables.py
Output:                 sonifold_paper/supp_tables/*.tex
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
RESULTS = root / "data" / "results"
REVISION = root / "data" / "experiments" / "revision"
OUT = root / "sonifold_paper" / "supp_tables"

# Display names for meshes, so the supplementary reads like the paper rather than
# like the filesystem.
MESH_LABEL = {
    "sphere": "Sphere", "sphere_genus0": "Sphere",
    "torus": "Torus", "torus_genus1": "Torus",
    "double_torus": "Double torus", "double_torus_genus2": "Double torus",
    "triple_torus_genus3": "Genus 3", "quad_torus_genus4": "Genus 4",
    "penta_torus_genus5": "Genus 5", "hex_torus_genus6": "Genus 6",
    "cube": "Cube", "tetrahedron": "Tetrahedron", "octahedron": "Octahedron",
    "icosahedron": "Icosahedron", "ellipsoid": "Ellipsoid",
    "flat_plate": "Flat plate", "rounded_cube": "Rounded cube",
    "genus2_asymmetric": "Genus 2 (asymmetric)", "genus7": "Genus 7", "genus8": "Genus 8",
    "torus_bent": "Bent torus", "torus_Rr2": "Torus $(R/r=2)$",
    "torus_Rr3": "Torus $(R/r=3)$", "torus_Rr5": "Torus $(R/r=5)$",
    "ellipsoid_112": "Ellipsoid $(1{:}1{:}2)$", "ellipsoid_115": "Ellipsoid $(1{:}1{:}5)$",
    "ellipsoid_123": "Ellipsoid $(1{:}2{:}3)$",
}
STRATEGY_LABEL = {"direct": "Direct", "mel": "Mel", "energy": "Energy"}


def tex_escape(s) -> str:
    """Escape LaTeX specials in a cell that came from the data, not from us."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def label(mesh: str) -> str:
    return MESH_LABEL.get(mesh, tex_escape(mesh))


def read_csv(name: str) -> list[dict]:
    path = RESULTS / name if not name.startswith("/") else Path(name)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=2, dash="--"):
    """Format a number for a table cell; blanks and unparseables become an en-dash."""
    if x is None or x == "":
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return tex_escape(x)
    if nd == 0:
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"  wrote {name}")


def tabular(colspec: str, header: list[str], rows: list[list[str]]) -> str:
    """A booktabs tabular. The caller owns the surrounding table environment."""
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- mesh quality

def build_mesh_quality():
    # Faces (approximately twice the vertex count) and the aspect median are dropped
    # so the table fits the column width; the mean and max aspect carry the quality signal.
    rows = []
    for r in read_csv("mesh_quality_report.csv"):
        rows.append([
            label(r["mesh"]), r["genus"], f"{int(r['n_vertices']):,}",
            fnum(r["aspect_mean"]), fnum(r["aspect_max"], 1),
            fnum(r["area_cv"]), fnum(r["edge_cv"]),
            r["euler"], r["euler_expected"],
            r"\checkmark" if r["euler_ok"].lower() == "true" else r"$\times$",
        ])
    write("mesh_quality.tex", tabular(
        "llrrrrrrrc",
        ["Mesh", "$g$", "Vertices", r"Aspect $\bar{}$",
         "Aspect max", "Area CV", "Edge CV", r"$\chi$", r"$\chi$ exp.", "OK"],
        rows))


def build_cotangent():
    # Drop the redundant positive-side columns (n_positive = total - n_negative,
    # pct_positive = 100 - pct_negative) so the table fits the column width.
    recs = read_csv("cotangent_weight_negativity.csv")
    rows = [[label(r["mesh_id"]), r["genus"], f"{int(r['n_vertices']):,}",
             f"{int(float(r['total_off_diagonal'])):,}",
             f"{int(float(r['n_negative'])):,}", fnum(r["pct_negative"], 3),
             fnum(r["max_neg_magnitude"], 4)] for r in recs]
    write("cotangent_negativity.tex", tabular(
        "llrrrrr",
        ["Mesh", "$g$", "Vertices", "Off-diag.", "Negative", "\\% neg.",
         "Max neg. mag."], rows))


def build_convergence():
    src = RESULTS / "eigenvalue_convergence_test_all_genus.csv"
    if not src.exists():
        return
    # The CSV carries all 50 eigenvalues per row; the convergence claim is about H,
    # so the table reports the summary columns and drops the per-lambda dump.
    recs = read_csv("eigenvalue_convergence_test_all_genus.csv")
    rows = [[r["genus"], tex_escape(r["resolution"].replace("_", " ")),
             f"{int(r['num_vertices']):,}", fnum(r["mean_aspect_ratio"], 2),
             fnum(r["spectral_entropy_H"], 4),
             f'{fnum(r["mean_beta0"])} $\\pm$ {fnum(r["std_beta0"])}'] for r in recs]
    write("convergence_all_genus.tex", tabular(
        "llrrrr",
        ["$g$", "Resolution", "Vertices", r"Aspect $\bar{}$", "$H$", r"$\bar\beta_0$"],
        rows))


def build_octahedron_symmetry():
    recs = [r for r in read_csv("results_systematic.csv") if r["mesh_id"].startswith("octahedron_sym")]
    rows = [[r["asymmetry_level"], fnum(r["beta0"], 0), fnum(r["A_ratio"], 4), fnum(r["S"], 4)]
            for r in recs]
    write("octahedron_symmetry.tex", tabular(
        "lrrr", [r"$z$-stretch", r"$\beta_0$", r"$A_{\mathrm{ratio}}$", "$S$"], rows))


# ---------------------------------------------------------------- main results

def build_full_results():
    """The 189-condition table the paper names, split by strategy for legibility."""
    recs = read_csv("results.csv")
    order = ["sphere", "torus", "double_torus", "cube", "tetrahedron",
             "octahedron", "icosahedron", "ellipsoid", "flat_plate"]
    audios = sorted({r["audio"] for r in recs})
    for strat in ["direct", "mel", "energy"]:
        by = {(r["mesh"], r["audio"]): r for r in recs if r["strategy"] == strat}
        rows = []
        for m in order:
            for a in audios:
                r = by.get((m, a))
                if not r:
                    continue
                rows.append([label(m) if a == audios[0] else "", a,
                             fnum(r["beta0"], 0), fnum(r["beta1"], 0), fnum(r["chi"], 0),
                             fnum(r["A_ratio"], 4), fnum(r["S"], 4)])
        write(f"full_results_{strat}.tex", tabular(
            "llrrrrr",
            ["Mesh", "Stimulus", r"$\beta_0$", r"$\beta_1$", r"$\chi$",
             r"$A_{\mathrm{ratio}}$", "$S$"], rows))
    write("full_results_count.tex", f"{len(recs)}\n")


def build_mapping_comparison():
    recs = read_csv("mapping_comparison.csv")
    rows = []
    seen = set()
    for r in recs:
        key = (r["mesh"], r["audio"])
        first = key not in seen
        seen.add(key)
        rows.append([
            f'{label(r["mesh"])} $\\times$ {r["audio"]}' if first else "",
            STRATEGY_LABEL.get(r["mapping"], r["mapping"]),
            f'{fnum(r["beta0_mean"])} $\\pm$ {fnum(r["beta0_std"])}',
            f'{fnum(r["A_ratio_mean"], 4)} $\\pm$ {fnum(r["A_ratio_std"], 4)}',
            f'{fnum(r["S_mean"], 4)} $\\pm$ {fnum(r["S_std"], 4)}',
        ])
    write("mapping_comparison.tex", tabular(
        "llrrr",
        ["Pair", "Strategy", r"$\bar\beta_0$", r"$\bar A_{\mathrm{ratio}}$", r"$\bar S$"],
        rows))


def build_random_vs_audio():
    recs = read_csv("random_vs_audio_comparison.csv")
    audios = [k[len("beta0_"):] for k in recs[0] if k.startswith("beta0_A")]
    rows = []
    for r in recs:
        rows.append([label(r["mesh"]),
                     f'{fnum(r["beta0_random_mean"])} $\\pm$ {fnum(r["beta0_random_std"])}']
                    + [fnum(r[f"beta0_{a}"]) for a in audios])
        rows.append(["", "$z$"] + [fnum(r[f"z_{a}"]) for a in audios])
    write("random_vs_audio.tex", tabular(
        "ll" + "r" * len(audios),
        ["Mesh", "Random baseline"] + audios, rows))


# ---------------------------------------------------------------- E1-E4 ablations

def _revision(name):
    with open(REVISION / name, encoding="utf-8") as f:
        return json.load(f)


def build_E1():
    d = _revision("E1_rescale.json")
    rows = []
    for rec in sorted(d["records"], key=lambda r: (r["shape"], r["audio"], r["alpha"])):
        rows.append([label(rec["shape"]), rec["audio"], fnum(rec["alpha"], 1),
                     fnum(rec["beta0_mean"], 6), fnum(rec["beta0_std"], 6),
                     fnum(rec["A_ratio_mean"], 6), fnum(rec["lambda_max"], 4)])
    write("E1_rescale.tex", tabular(
        "llrrrrr",
        ["Shape", "Stimulus", r"$\alpha$", r"$\bar\beta_0$", r"$\sigma(\beta_0)$",
         r"$\bar A_{\mathrm{ratio}}$", r"$\lambda_{\max}$"], rows))


def build_E2():
    d = _revision("E2_transpose.json")
    shapes = d["shapes"]
    semis = d["semitones"]
    by = {(r["shape"], r["semitones"]): r for r in d["records"]}
    rows = []
    for sh in shapes:
        rows.append([label(sh)] + [fnum(by[(sh, s)]["beta0_mean"]) if (sh, s) in by else "--"
                                   for s in semis])
    write("E2_transpose.tex", tabular(
        "l" + "r" * len(semis),
        ["Shape"] + [f"${s:+d}$" for s in semis], rows))


def build_E3():
    d = _revision("E3_freqmatch.json")
    by = {(r["shape"], r["audio"], r["strategy"]): r for r in d["records"]}
    rows = []
    for sh in d["shapes"]:
        for a in d["stimuli"]:
            cells = []
            for st in d["strategies"]:
                r = by.get((sh, a, st))
                cells.append(f'{fnum(r["beta0_mean"])} $\\pm$ {fnum(r["beta0_std"])}' if r else "--")
            rows.append([label(sh) if a == d["stimuli"][0] else "", a] + cells)
    write("E3_freqmatch.tex", tabular(
        "ll" + "r" * len(d["strategies"]),
        ["Shape", "Stimulus"] + [STRATEGY_LABEL.get(s, s) for s in d["strategies"]], rows))


def build_E4():
    d = _revision("E4_sameg.json")
    rows = []
    for rec in sorted(d["records"], key=lambda r: (r["label"], r["audio"])):
        rows.append([tex_escape(rec["label"]),
                     f'({fnum(rec["R"], 1)}, {fnum(rec["r"], 1)})', tex_escape(rec["audio"]),
                     f'{fnum(rec["beta0_mean"])} $\\pm$ {fnum(rec["beta0_std"])}',
                     fnum(rec["A_ratio_mean"], 4)])
    write("E4_sameg.tex", tabular(
        "lllrr",
        ["Embedding", "$(R,r)$", "Stimulus", r"$\bar\beta_0$", r"$\bar A_{\mathrm{ratio}}$"],
        rows))


# ---------------------------------------------------------------- genus, entropy

def build_genus_sequence():
    recs = read_csv("results_genus_extended_7point.csv")
    rows = [[label(r["mesh_id"]), r["genus"], fnum(r["beta0"], 0),
             fnum(r["A_ratio"], 4), fnum(r["S"], 4)] for r in recs]
    write("genus_7point.tex", tabular(
        "llrrr", ["Mesh", "$g$", r"$\beta_0$", r"$A_{\mathrm{ratio}}$", "$S$"], rows))


def build_multiplicity():
    recs = read_csv("eigenvalue_multiplicity_analysis.csv")
    rows = [[label(r["mesh"]), r["genus"], r["M_eff_clusters"],
             fnum(r["M_eff_mean_size"]), fnum(r["spectral_entropy_H"], 4),
             fnum(r["spectral_gap_gamma"], 4), fnum(r["eigenvalue_density_rho"], 4),
             fnum(r["product_score_P"], 4)] for r in recs]
    write("multiplicity.tex", tabular(
        "llrrrrrr",
        ["Mesh", "$g$", "Clusters", "Mean size", "$H$", r"$\gamma$", r"$\rho$", "$P$"],
        rows))


def build_K_sensitivity():
    recs = read_csv("K_sensitivity_genus_comparison.csv")
    rows = [[label(r["mesh"]), r["genus"], fnum(r["beta0_K50"]), fnum(r["beta0_K100"]),
             fnum(r["beta0_K200"])] for r in recs]
    write("K_sensitivity.tex", tabular(
        "llrrr", ["Mesh", "$g$", "$K=50$", "$K=100$", "$K=200$"], rows))


def build_entropy_vs_beta0():
    recs = read_csv("extended_mesh_spectral_entropy.csv")
    # genus2_asymmetric ships as a byte-identical copy of the 4846-vertex double_torus
    # (same vertices and eigenvalues), so it would render as a duplicate row. Drop it.
    recs = [r for r in recs if r["mesh_name"] != "genus2_asymmetric"]
    rows = [[label(r["mesh_name"]), r["genus"], f'{int(r["num_vertices"]):,}',
             fnum(r["spectral_entropy_H"], 4),
             f'{fnum(r["mean_beta0_random"])} $\\pm$ {fnum(r["std_beta0_random"])}',
             fnum(r["mean_beta0_A5"])] for r in recs]
    write("entropy_vs_beta0.tex", tabular(
        "llrrrr",
        ["Mesh", "$g$", "Vertices", "$H$", r"$\bar\beta_0$ (random)", r"$\bar\beta_0$ (A5)"],
        rows))


def build_nazarov_sodin():
    recs = read_csv("nazarov_sodin_comparison.csv")
    rows = [[r["L"], r["L_squared"],
             f'{fnum(r["mean_beta0"])} $\\pm$ {fnum(r["std_beta0"])}'] for r in recs]
    write("nazarov_sodin.tex", tabular(
        "rrr", ["$\\ell$", "$\\ell^2$", r"$\bar\beta_0$"], rows))


def build_epsilon_robustness():
    src = RESULTS / "epsilon_robustness_correlations.csv"
    if not src.exists():
        return
    recs = read_csv("epsilon_robustness_correlations.csv")
    keys = list(recs[0].keys())
    rows = [[fnum(r[k], 4) if k != keys[0] else tex_escape(r[k]) for k in keys] for r in recs]
    write("epsilon_robustness.tex", tabular(
        "l" + "r" * (len(keys) - 1),
        [tex_escape(k.replace("_", " ")) for k in keys], rows))


# ---------------------------------------------------------------- temporal

def build_autocorrelation():
    recs = read_csv("temporal_autocorrelation.csv")
    rows = [[label(r["mesh"].lower().replace(" ", "_")), r["audio"], r["n_frames"],
             f'{fnum(r["mean_beta0"])} $\\pm$ {fnum(r["std_beta0"])}',
             fnum(r["lag1_autocorrelation"], 3), fnum(r["n_effective"], 1),
             fnum(r["naive_SE"], 3), fnum(r["corrected_SE"], 3)] for r in recs]
    write("autocorrelation.tex", tabular(
        "llrrrrrr",
        ["Mesh", "Stimulus", "$N$", r"$\bar\beta_0$", "$r_1$",
         r"$N_{\mathrm{eff}}$", "Naive SE", "Corrected SE"], rows))


def build_temporal_correlations():
    recs = read_csv("temporal_correlations.csv")
    keys = list(recs[0].keys())
    rows = [[label(r[keys[0]])] + [fnum(r[k], 3) for k in keys[1:]] for r in recs]
    write("temporal_correlations.tex", tabular(
        "l" + "r" * (len(keys) - 1),
        [tex_escape(k.replace("_", " ")) for k in keys], rows))


def build_temporal_genus():
    recs = read_csv("temporal_genus_summary.csv")
    rows = [[label(r["mesh_name"]), r["genus"], fnum(r["mean_beta0"]), fnum(r["std_beta0"]),
             fnum(r["cv_beta0"], 3), fnum(r["mean_delta_beta0"]), fnum(r["max_delta_beta0"], 0)]
            for r in recs]
    write("temporal_genus.tex", tabular(
        "llrrrrr",
        ["Mesh", "$g$", r"$\bar\beta_0$", r"$\sigma(\beta_0)$", "CV",
         r"$\overline{|\Delta\beta_0|}$", r"$\max|\Delta\beta_0|$"], rows))


# ---------------------------------------------------------------- sequencing

def build_sequencing():
    for src, out, tcol in [("shape_sequencing_data.csv", "shape_sequencing.tex", "frame_index"),
                           ("cello_sequencing_data.csv", "cello_sequencing.tex", "time_sec")]:
        recs = read_csv(src)
        has_S = "S" in recs[0]
        rows = []
        seen = set()
        for r in recs:
            k = r[tcol]
            first = k not in seen
            seen.add(k)
            cells = [tex_escape(k) if first else "", fnum(r["rms"], 5) if first else "",
                     label(r["shape"]), fnum(r["beta0"], 0),
                     fnum(r.get("a_ratio", r.get("A_ratio")), 4)]
            if has_S:
                cells.append(fnum(r["S"], 4))
            rows.append(cells)
        header = ["Frame" if tcol == "frame_index" else "Time (s)", "RMS", "Shape",
                  r"$\beta_0$", r"$A_{\mathrm{ratio}}$"] + ([r"$S$"] if has_S else [])
        write(out, tabular("llr" + "r" * (2 + has_S), header, rows))


def main():
    print(f"Building supplementary tables -> {OUT}")
    for fn in [build_mesh_quality, build_cotangent, build_convergence,
               build_octahedron_symmetry, build_full_results, build_mapping_comparison,
               build_random_vs_audio, build_E1, build_E2, build_E3, build_E4,
               build_genus_sequence, build_multiplicity, build_K_sensitivity,
               build_entropy_vs_beta0, build_nazarov_sodin, build_epsilon_robustness,
               build_autocorrelation, build_temporal_correlations, build_temporal_genus,
               build_sequencing]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  SKIP {fn.__name__}: missing {e.filename}")
    print("done.")


if __name__ == "__main__":
    main()
