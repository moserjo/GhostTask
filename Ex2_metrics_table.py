"""
Ex2_metrics_table.py
--------------------
Builds the LaTeX results tables for Experiment 2 (Tripendulum).

Experiment 2 has THREE inferred quantities, so THREE tables are produced per
comparison (one per parameter).  Two comparisons are made, giving SIX tables
total, all written one after another into Ex2_table.txt:

  (A)  standard / extra_npoints / extra_npoints_extra_sigma
       For PIGP only `standard` is shown (the extra-point modes leave PIGP's
       training inputs unchanged, so they are dropped from the PIGP block).
  (B)  known_u / known_u_known_sigma   (both methods, both modes)

For every combination of
    npoints  in {3, 5, 10, 25}
    sigma    in {0.001, 0.01, 0.1, 0.2}   (file label uses %.2f -> 0.00/0.01/0.10/0.20)
    method   in {GT, PIGP}
it reads the first 10 runs (run = 0..9) and computes, from
learned_parameters = [l, g]  and  true_parameters = [l_true, g_true], the
relative error of each parameter:

    param 0 :  |Delta l / l|       = |(l - l_true) / l_true|
    param 1 :  |Delta g / g|       = |(g - g_true) / g_true|
    param 2 :  |Delta (l/g)/(l/g)| = |((l/g) - (l_true/g_true)) / (l_true/g_true)|

Outliers are removed EXACTLY as seaborn's boxplot does (matplotlib's 1.5*IQR
whisker rule).  In each row the lowest mean is bold; adjacent data-mode cells
whose raw per-run data coincide are merged (centered).  The analysed parameter
is printed in the top-left corner of every table.

Output: Ex2_table.txt
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

# display names for the data-mode column headers
MODE_DISPLAY = {
    "standard": "standard",
    "extra_npoints": r"extra\_n",
    "extra_npoints_extra_sigma": r"extra\_n+$\sigma$",
    "known_u": r"known\_u",
    "known_u_known_sigma": r"known\_u+$\sigma$",
}

# LaTeX label of each analysed parameter (corner cell + caption)
PARAM_LABEL = {
    0: r"$\frac{\Delta \ell}{\ell}$",
    1: r"$\frac{\Delta g}{g}$",
    2: r"$\frac{\Delta \ell/g}{\ell/g}$",
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


def snum(value):
    """Plain number string; siunitx handles rounding + scientific notation."""
    return f"{value:.6g}"


def s_cell(value, method_min, overall_min, method_multi):
    """Bare S-column cell; overall-best -> bold, per-method best -> italic."""
    if _is_nan(value):
        return "{}"
    body = snum(value)
    if overall_min is not None and value == overall_min:
        return r"\bfseries " + body
    if method_multi and method_min is not None and value == method_min:
        return r"\itshape " + body
    return body


def num_cell(value, method_min, overall_min, method_multi):
    r"""\num{} cell for \multicolumn spans (which drop the S column type)."""
    if _is_nan(value):
        return ""
    body = r"\num{" + snum(value) + "}"
    if overall_min is not None and value == overall_min:
        return r"{\bfseries " + body + "}"
    if method_multi and method_min is not None and value == method_min:
        return r"{\itshape " + body + "}"
    return body


def group_modes(arrays):
    """
    Group a method's data-mode columns into spans of *coinciding* data:
    consecutive columns whose raw per-run error arrays are identical are merged
    into a single (start, length) span.  Returns a list of (start, length).
    """
    groups = []
    i = 0
    while i < len(arrays):
        j = i + 1
        while (j < len(arrays)
               and arrays[i].size > 0
               and np.array_equal(arrays[i], arrays[j])):
            j += 1
        groups.append((i, j - i))
        i = j
    return groups


def render_region(values, groups, method_min, overall_min, method_multi):
    """
    Render one method's S-column region, honouring merged spans (booktabs, no
    vertical rules).  overall-best cell -> bold, per-method best -> italic.
    """
    tokens = []
    for start, length in groups:
        val = values[start]                      # coinciding -> any member works
        if length > 1:                            # merged span -> centered \num
            body = num_cell(val, method_min, overall_min, method_multi)
            tokens.append(rf"\multicolumn{{{length}}}{{c}}{{{body}}}")
        else:                                     # single -> bare S cell
            tokens.append(s_cell(val, method_min, overall_min, method_multi))
    return " & ".join(tokens)


def header_region(modes):
    """Data-mode names for one method's S-column region (braced = text cells)."""
    return " & ".join("{" + MODE_DISPLAY[m] + "}" for m in modes)


def build_table(param_idx, gt_modes, pigp_modes, caption):
    """Build one full LaTeX table (list of lines) for a parameter / comparison."""
    method_modes = {"GT": gt_modes, "PIGP": pigp_modes}

    # compute raw per-run arrays + outlier-filtered means for every cell
    raw_errs = {}
    means = {}
    for n_idx, npoints in enumerate(n_list):
        for s_idx, (s_label, s_file) in enumerate(sigma_list):
            cell_raw = {m: {} for m in methods}
            cell_mean = {m: {} for m in methods}
            for method in methods:
                for dmode in method_modes[method]:
                    errs = relative_errors(npoints, s_file, method, dmode, param_idx)
                    cell_raw[method][dmode] = errs
                    cell_mean[method][dmode] = mean_without_outliers(errs)
            raw_errs[(n_idx, s_idx)] = cell_raw
            means[(n_idx, s_idx)] = cell_mean

    corner = PARAM_LABEL[param_idx]
    n_gt, n_pigp = len(gt_modes), len(pigp_modes)
    spec = "ll" + "S" * n_gt + "S" * n_pigp
    gt_lo, gt_hi = 3, 2 + n_gt                    # \cmidrule column ranges
    pg_lo, pg_hi = 3 + n_gt, 2 + n_gt + n_pigp
    header = [
        r"% Requires \usepackage{booktabs} and \usepackage{siunitx} in the preamble.",
        r"\begin{table}[]",
        r"\sisetup{round-mode=figures, round-precision=2, scientific-notation=true, "
        r"table-format=1.2e-1, detect-weight=true, detect-shape=true, detect-family=true}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        rf"{{{corner}}} & & \multicolumn{{{n_gt}}}{{c}}{{\textbf{{GT}}}} "
        rf"& \multicolumn{{{n_pigp}}}{{c}}{{PIGP}} \\",
        rf"\cmidrule(lr){{{gt_lo}-{gt_hi}}} \cmidrule(lr){{{pg_lo}-{pg_hi}}}",
        rf"n & {{$\sigma$}} & {header_region(gt_modes)} & {header_region(pigp_modes)} \\",
        r"\midrule",
    ]
    footer = [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"]

    rows = []
    for n_idx, npoints in enumerate(n_list):
        for s_idx, (s_label, s_file) in enumerate(sigma_list):
            if s_idx == 0 and n_idx > 0:
                rows.append(r"\addlinespace")
            cell_mean = means[(n_idx, s_idx)]
            cell_raw = raw_errs[(n_idx, s_idx)]

            groups = {m: group_modes([cell_raw[m][d] for d in method_modes[m]])
                      for m in methods}

            # per-method best (one representative value per rendered/merged cell)
            method_min = {}
            for m in methods:
                vals = [cell_mean[m][d] for d in method_modes[m]]
                reps = [vals[start] for start, _ in groups[m] if not _is_nan(vals[start])]
                method_min[m] = min(reps) if reps else None
            valid = [v for v in method_min.values() if v is not None]
            overall_min = min(valid) if valid else None

            gt = render_region([cell_mean["GT"][d] for d in gt_modes], groups["GT"],
                               method_min["GT"], overall_min, len(gt_modes) >= 2)
            pigp = render_region([cell_mean["PIGP"][d] for d in pigp_modes], groups["PIGP"],
                                 method_min["PIGP"], overall_min, len(pigp_modes) >= 2)

            rows.append(rf"{npoints} & {s_label} & {gt} & {pigp} \\")

    # overall error per data mode: geometric mean of cell means over all n, sigma
    def _gmean(vals):
        vals = [v for v in vals if not _is_nan(v) and v > 0]
        return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")

    agg = {m: {d: _gmean([means[(ni, si)][m][d]
                          for ni in range(len(n_list)) for si in range(len(sigma_list))])
               for d in method_modes[m]} for m in methods}
    agg_mmin = {m: min([agg[m][d] for d in method_modes[m] if not _is_nan(agg[m][d])],
                       default=None) for m in methods}
    _valid = [v for v in agg_mmin.values() if v is not None]
    agg_omin = min(_valid) if _valid else None
    gt_sum = render_region([agg["GT"][d] for d in gt_modes],
                           [(i, 1) for i in range(len(gt_modes))],
                           agg_mmin["GT"], agg_omin, len(gt_modes) >= 2)
    pigp_sum = render_region([agg["PIGP"][d] for d in pigp_modes],
                             [(i, 1) for i in range(len(pigp_modes))],
                             agg_mmin["PIGP"], agg_omin, len(pigp_modes) >= 2)
    rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{overall}}}} & {gt_sum} & {pigp_sum} \\")

    return header + rows + footer


# --------------------------------------------------------------------------- #
# build all six tables and write them out
# --------------------------------------------------------------------------- #
# Comparison A: standard / extra_npoints / extra_npoints_extra_sigma
#               (PIGP block: standard only)
gt_modes_A = ["standard", "extra_npoints", "extra_npoints_extra_sigma"]
pigp_modes_A = ["standard"]

# Comparison B: known_u / known_u_known_sigma (both methods)
modes_B = ["known_u", "known_u_known_sigma"]

blocks = []
for param_idx in (0, 1, 2):
    caption = f"Relative error {PARAM_LABEL[param_idx]}"
    print(f"\n--- table A, param {param_idx}: {caption} ---")
    blocks.append("\n".join(build_table(param_idx, gt_modes_A, pigp_modes_A, caption)))

for param_idx in (0, 1, 2):
    caption = rf"Relative error {PARAM_LABEL[param_idx]} (known $u$)"
    print(f"--- table B, param {param_idx}: {caption} ---")
    blocks.append("\n".join(build_table(param_idx, modes_B, modes_B, caption)))

table = "\n\n".join(blocks) + "\n"

out_path = os.path.join(base_dir, "Ex2_table.txt")
with open(out_path, "w") as f:
    f.write(table)

print(f"\nWrote 6 tables to {out_path}\n")
print(table)
