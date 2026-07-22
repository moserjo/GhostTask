"""
Ex2_metrics_table_combined.py
-----------------------------
Alternative renderer of the Experiment 2 (Tripendulum) LaTeX results tables.

Unlike Ex2_metrics_table.py (which produces one table per inferred parameter,
six tables in total), this version packs ALL THREE inferred quantities into a
single cell, written as

        param0 / param1 / param2      e.g.   0.1/0.2/0.01

with
    param 0 :  |Delta l / l|          = |(l - l_true) / l_true|
    param 1 :  |Delta g / g|          = |(g - g_true) / g_true|
    param 2 :  |Delta (l/g)/(l/g)|    = |((l/g) - (l_true/g_true)) / (l_true/g_true)|

Only the two data-mode COMPARISONS remain separate tables (they involve
different data modes), so TWO tables are produced, one after another, into
Ex2_table_combined.txt:

  (A)  standard / extra_npoints / extra_npoints_extra_sigma
       For PIGP only `standard` is shown (the extra-point modes leave PIGP's
       training inputs unchanged, so they are dropped from the PIGP block).
  (B)  known_u / known_u_known_sigma   (both methods, both modes)

Metric, outlier handling (matplotlib 1.5*IQR whisker rule), overall-best (bold)
and per-method-best (italic) highlighting, and merging of adjacent data-mode
cells whose raw per-run data coincide are all identical to Ex2_metrics_table.py.
Highlighting is applied PER PARAMETER inside each combined cell, so the three
slash-separated numbers are styled independently.

Output: Ex2_table_combined.txt
"""

import os
import numpy as np
import matplotlib.cbook as cbook

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
base_dir = os.path.dirname(os.path.abspath(__file__))
output_data_ex2 = os.path.join(
    base_dir, "Experiment2_Tripendulum", "output_data"
)

runs = 10                                   # first 10 runs: 0 .. runs-1

n_list = [3, 5, 10, 25]                      # table rows (npoints)
# (display label for the row, sigma value used in the filename via %.2f)
sigma_list = [("0.001", 0.00),
              ("0.01", 0.01),
              ("0.1", 0.10),
              ("0.2", 0.20)]

methods = ["GT", "PIGP"]

# the three parameters packed into every cell, in display order
PARAMS = (0, 1, 2)

# display names for the data-mode column headers
MODE_DISPLAY = {
    "standard": "standard",
    "extra_npoints": r"extra\_n",
    "extra_npoints_extra_sigma": r"extra\_n+$\sigma$",
    "known_u": r"known\_u",
    "known_u_known_sigma": r"known\_u+$\sigma$",
}

# LaTeX label of each analysed parameter (used in the corner cell)
PARAM_LABEL = {
    0: r"\frac{\Delta \ell}{\ell}",
    1: r"\frac{\Delta g}{g}",
    2: r"\frac{\Delta \ell/g}{\ell/g}",
}

NUM_FMT = "{:.2g}"                           # 2 significant figures


# --------------------------------------------------------------------------- #
# data loading + metric
# --------------------------------------------------------------------------- #
def _rel_error(learned, true, param_idx):
    """Relative error of the requested parameter (0:l, 1:g, 2:ratio l/g)."""
    if param_idx == 0:
        return abs((learned[0] - true[0]) / true[0])
    if param_idx == 1:
        return abs((learned[1] - true[1]) / true[1])
    learned_ratio = learned[0] / learned[1]
    true_ratio = true[0] / true[1]
    return abs((learned_ratio - true_ratio) / true_ratio)


def relative_errors(npoints, sigma_file, method, data_mode, param_idx):
    """Per-run relative errors of `param_idx` for the first `runs` runs."""
    data_dir = os.path.join(output_data_ex2, data_mode)
    errs = []
    for run in range(runs):
        fname = f"Ex2_{method}_n{npoints}_sigma{sigma_file:.2f}_{run}.npz"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] missing file: {os.path.relpath(path, base_dir)}")
            continue
        raw = np.load(path, allow_pickle=True)
        learned = np.array(raw["learned_parameters"]).reshape(-1)
        true = np.array(raw["true_parameters"]).reshape(-1)
        errs.append(float(_rel_error(learned, true, param_idx)))
    return np.asarray(errs, dtype=float)


def mean_without_outliers(values):
    """
    Mean of `values` after dropping outliers exactly as seaborn/matplotlib
    boxplot does: points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (the whisker
    fences computed by matplotlib.cbook.boxplot_stats) are treated as fliers
    and removed.
    """
    if values.size == 0:
        return np.nan
    stats = cbook.boxplot_stats(values)[0]
    lo, hi = stats["whislo"], stats["whishi"]
    inliers = values[(values >= lo) & (values <= hi)]
    if inliers.size == 0:
        return np.nan
    return float(inliers.mean())


# --------------------------------------------------------------------------- #
# table rendering helpers
# --------------------------------------------------------------------------- #
def _is_nan(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def fmt_num(value):
    """2 significant figures; render e-notation as a \\times 10^{-x} (math)."""
    s = NUM_FMT.format(value)
    if "e" in s:
        mant, exp = s.split("e")
        s = rf"{mant} \times 10^{{{int(exp)}}}"
    return s


def combined_cell_tex(param_vals, method_mins, overall_mins, method_multi):
    r"""
    Render one cell holding all three parameters as `p0/p1/p2`, each number
    typeset with \num{} (siunitx handles rounding + scientific notation).  Each
    parameter is styled independently:
      overall-best -> bold, per-method best -> italic (only if >1 data mode).
    """
    parts = []
    for p in PARAMS:
        value = param_vals[p]
        if _is_nan(value):
            parts.append("-")
            continue
        body = r"\num{" + f"{value:.6g}" + "}"
        if overall_mins[p] is not None and value == overall_mins[p]:
            body = r"{\bfseries " + body + "}"
        elif method_multi and method_mins[p] is not None and value == method_mins[p]:
            body = r"{\itshape " + body + "}"
        parts.append(body)
    return "/".join(parts)


def _modes_equal(a, b):
    """True if two modes coincide across ALL three parameters' raw run data."""
    return all(a[p].size > 0 and np.array_equal(a[p], b[p]) for p in PARAMS)


def group_modes(arrays_by_mode):
    """
    Group a method's data-mode columns into spans of *coinciding* data:
    consecutive columns whose raw per-run error arrays are identical (for every
    parameter) are merged into a single (start, length) span.
    `arrays_by_mode` is a list over modes, each a dict {param_idx: np.array}.
    Returns a list of (start, length).
    """
    groups = []
    i = 0
    while i < len(arrays_by_mode):
        j = i + 1
        while j < len(arrays_by_mode) and _modes_equal(arrays_by_mode[i],
                                                        arrays_by_mode[j]):
            j += 1
        groups.append((i, j - i))
        i = j
    return groups


def render_region(mode_means, groups, method_mins, overall_mins, method_multi):
    """
    Render one method's column region as LaTeX, honouring merged spans (booktabs,
    no vertical rules).  `mode_means` is a list over modes of {param_idx: mean}.
    """
    tokens = []
    for start, length in groups:
        vals = mode_means[start]                 # coinciding -> any member works
        s = combined_cell_tex(vals, method_mins, overall_mins, method_multi)
        if length > 1:                            # merged span -> centered
            tokens.append(rf"\multicolumn{{{length}}}{{c}}{{{s}}}")
        else:
            tokens.append(s)
    return " & ".join(tokens)


def header_region(modes):
    """Render the data-mode names for one method's column region (no merges)."""
    return " & ".join(MODE_DISPLAY[m] for m in modes)


def _row_extrema(mode_means, groups):
    """
    For one method in one row: per-parameter minimum over the (merged) cells.
    Returns {param_idx: min_value_or_None}.
    """
    out = {}
    for p in PARAMS:
        reps = [mode_means[start][p] for start, _ in groups
                if not _is_nan(mode_means[start][p])]
        out[p] = min(reps) if reps else None
    return out


def build_table(gt_modes, pigp_modes, caption):
    """Build one full LaTeX table (list of lines) for a data-mode comparison."""
    method_modes = {"GT": gt_modes, "PIGP": pigp_modes}

    # compute raw per-run arrays + outlier-filtered means for every cell/param
    raw_errs = {}
    means = {}
    for n_idx, npoints in enumerate(n_list):
        for s_idx, (s_label, s_file) in enumerate(sigma_list):
            cell_raw = {m: {} for m in methods}
            cell_mean = {m: {} for m in methods}
            for method in methods:
                for dmode in method_modes[method]:
                    raw_p = {}
                    mean_p = {}
                    for p in PARAMS:
                        errs = relative_errors(npoints, s_file, method, dmode, p)
                        raw_p[p] = errs
                        mean_p[p] = mean_without_outliers(errs)
                    cell_raw[method][dmode] = raw_p
                    cell_mean[method][dmode] = mean_p
            raw_errs[(n_idx, s_idx)] = cell_raw
            means[(n_idx, s_idx)] = cell_mean

    corner = "$" + "/".join(PARAM_LABEL[p] for p in PARAMS) + "$"
    n_gt, n_pigp = len(gt_modes), len(pigp_modes)
    spec = "ll" + "c" * n_gt + "c" * n_pigp
    gt_lo, gt_hi = 3, 2 + n_gt                    # \cmidrule column ranges
    pg_lo, pg_hi = 3 + n_gt, 2 + n_gt + n_pigp
    # three numbers per cell make this table very wide -> shrink to \textwidth
    header = [
        r"% Requires \usepackage{booktabs}, \usepackage{siunitx} and "
        r"\usepackage{graphicx} (for \resizebox) in the preamble.",
        r"\begin{table}[]",
        r"\sisetup{round-mode=figures, round-precision=2, scientific-notation=true, "
        r"detect-weight=true, detect-shape=true, detect-family=true}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        rf"{corner} & & \multicolumn{{{n_gt}}}{{c}}{{\textbf{{GT}}}} "
        rf"& \multicolumn{{{n_pigp}}}{{c}}{{PIGP}} \\",
        rf"\cmidrule(lr){{{gt_lo}-{gt_hi}}} \cmidrule(lr){{{pg_lo}-{pg_hi}}}",
        rf"n & $\sigma$ & {header_region(gt_modes)} & {header_region(pigp_modes)} \\",
        r"\midrule",
    ]
    footer = [r"\bottomrule", r"\end{tabular}%", r"}",
              rf"\caption{{{caption}}}", r"\end{table}"]

    rows = []
    for n_idx, npoints in enumerate(n_list):
        for s_idx, (s_label, s_file) in enumerate(sigma_list):
            if s_idx == 0 and n_idx > 0:
                rows.append(r"\addlinespace")
            cell_mean = means[(n_idx, s_idx)]
            cell_raw = raw_errs[(n_idx, s_idx)]

            groups = {m: group_modes([cell_raw[m][d] for d in method_modes[m]])
                      for m in methods}

            # per-method best per parameter (one representative per merged cell)
            method_min = {m: _row_extrema([cell_mean[m][d] for d in method_modes[m]],
                                          groups[m]) for m in methods}
            # overall best per parameter across both methods
            overall_min = {}
            for p in PARAMS:
                vals = [method_min[m][p] for m in methods
                        if method_min[m][p] is not None]
                overall_min[p] = min(vals) if vals else None

            gt = render_region([cell_mean["GT"][d] for d in gt_modes], groups["GT"],
                               method_min["GT"], overall_min, len(gt_modes) >= 2)
            pigp = render_region([cell_mean["PIGP"][d] for d in pigp_modes], groups["PIGP"],
                                 method_min["PIGP"], overall_min, len(pigp_modes) >= 2)

            rows.append(rf"{npoints} & {s_label} & {gt} & {pigp} \\")

    # overall error per data mode / parameter: geometric mean over all n, sigma
    def _gmean(vals):
        vals = [v for v in vals if not _is_nan(v) and v > 0]
        return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")

    agg = {m: {d: {p: _gmean([means[(ni, si)][m][d][p]
                              for ni in range(len(n_list))
                              for si in range(len(sigma_list))])
                   for p in PARAMS}
               for d in method_modes[m]} for m in methods}
    agg_groups = {m: [(i, 1) for i in range(len(method_modes[m]))] for m in methods}
    agg_mmin = {m: _row_extrema([agg[m][d] for d in method_modes[m]], agg_groups[m])
                for m in methods}
    agg_omin = {}
    for p in PARAMS:
        vals = [agg_mmin[m][p] for m in methods if agg_mmin[m][p] is not None]
        agg_omin[p] = min(vals) if vals else None

    gt_sum = render_region([agg["GT"][d] for d in gt_modes], agg_groups["GT"],
                           agg_mmin["GT"], agg_omin, len(gt_modes) >= 2)
    pigp_sum = render_region([agg["PIGP"][d] for d in pigp_modes], agg_groups["PIGP"],
                             agg_mmin["PIGP"], agg_omin, len(pigp_modes) >= 2)
    rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{overall}}}} & {gt_sum} & {pigp_sum} \\")

    return header + rows + footer


# --------------------------------------------------------------------------- #
# build the two tables and write them out
# --------------------------------------------------------------------------- #
# Comparison A: standard / extra_npoints / extra_npoints_extra_sigma
#               (PIGP block: standard only)
gt_modes_A = ["standard", "extra_npoints", "extra_npoints_extra_sigma"]
pigp_modes_A = ["standard"]

# Comparison B: known_u / known_u_known_sigma (both methods)
modes_B = ["known_u", "known_u_known_sigma"]

cell_note = (r"each cell: $\frac{\Delta \ell}{\ell}$ / $\frac{\Delta g}{g}$ / "
             r"$\frac{\Delta \ell/g}{\ell/g}$")

blocks = []
print("\n--- table A (standard / extra points) ---")
blocks.append("\n".join(
    build_table(gt_modes_A, pigp_modes_A, f"Relative errors ({cell_note})")))

print("--- table B (known $u$) ---")
blocks.append("\n".join(
    build_table(modes_B, modes_B, f"Relative errors, known $u$ ({cell_note})")))

table = "\n\n".join(blocks) + "\n"

out_path = os.path.join(base_dir, "Ex2_table_combined.txt")
with open(out_path, "w") as f:
    f.write(table)

print(f"\nWrote 2 tables to {out_path}\n")
print(table)
