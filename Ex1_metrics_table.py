"""
Ex1_metrics_table.py
--------------------
Builds the LaTeX results table for Experiment 1 (Pedagogical).

For every combination of
    npoints  in {3, 5, 10}
    sigma    in {0.001, 0.01, 0.1, 0.2}   (file label uses %.2f -> 0.00/0.01/0.10/0.20)
    method   in {GT, PIGP}
    data_mode in {standard, extra_npoints, extra_npoints_extra_sigma}
it reads the first 10 runs (run = 0..9), computes the relative error of the
learned parameter a (true value 2):

        rel_err = |a - 2| / 2

then removes outliers EXACTLY as seaborn's boxplot does (matplotlib's
1.5*IQR whisker rule, via matplotlib.cbook.boxplot_stats) and averages the
remaining runs.

The means are written into the table layout from template.txt. In each row
(one per npoints/sigma) the lowest mean value across the 6 method/data-mode
cells is marked in bold (\\textbf{...}).

Output: Ex1_table.txt  (copy-paste ready LaTeX).
"""

import os
import numpy as np
import matplotlib.cbook as cbook

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
base_dir = os.path.dirname(os.path.abspath(__file__))
output_data_ex1 = os.path.join(
    base_dir, "Experiment1_Pedagogical", "optimisation", "output_data"
)

runs = 10                                   # first 10 runs: 0 .. runs-1
true_a = 2.0

n_list = [3, 5, 10, 15]                      # table rows (npoints)
# (display label for the row, sigma value used in the filename via %.2f)
sigma_list = [("0.001", 0.00),
              ("0.01", 0.01),
              ("0.1", 0.10),
              ("0.2", 0.20)]

methods = ["GT", "PIGP"]                                     # column groups
data_modes = ["standard", "extra_npoints", "extra_npoints_extra_sigma"]

NUM_FMT = "{:.2g}"                           # 2 significant figures


# --------------------------------------------------------------------------- #
# data loading + metric
# --------------------------------------------------------------------------- #
def relative_errors(npoints, sigma_file, method, data_mode):
    """Relative errors |a-2|/2 of the learned parameter for the first `runs` runs."""
    data_dir = os.path.join(output_data_ex1, data_mode)
    errs = []
    for run in range(runs):
        fname = f"Ex1_n{npoints}_sigma{sigma_file:.2f}_{method}_{run}.npz"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] missing file: {os.path.relpath(path, base_dir)}")
            continue
        raw = np.load(path, allow_pickle=True)
        a = float(np.array(raw["learned_parameters"]).reshape(-1)[-1])
        errs.append(abs(a - true_a) / true_a)
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
# compute, for every cell, both the raw per-run error array (used to detect
# coinciding data modes -> merged cells) and the outlier-filtered mean.
#   raw_errs[(n_idx, s_idx)][method][data_mode]  -> np.ndarray
#   means[(n_idx, s_idx)][method][data_mode]     -> float
# --------------------------------------------------------------------------- #
raw_errs = {}
means = {}
for n_idx, npoints in enumerate(n_list):
    for s_idx, (s_label, s_file) in enumerate(sigma_list):
        cell_raw = {m: {} for m in methods}
        cell_mean = {m: {} for m in methods}
        for method in methods:
            for dmode in data_modes:
                errs = relative_errors(npoints, s_file, method, dmode)
                cell_raw[method][dmode] = errs
                cell_mean[method][dmode] = mean_without_outliers(errs)
        raw_errs[(n_idx, s_idx)] = cell_raw
        means[(n_idx, s_idx)] = cell_mean


# --------------------------------------------------------------------------- #
# build the LaTeX table
# --------------------------------------------------------------------------- #
# Header / preamble kept verbatim from template.txt
header = [
    r"% Requires \usepackage{booktabs} and \usepackage{siunitx} in the preamble.",
    r"\begin{table}[]",
    r"\sisetup{round-mode=figures, round-precision=2, scientific-notation=true, "
    r"table-format=1.2e-1, detect-weight=true, detect-shape=true, detect-family=true}",
    r"\begin{tabular}{llSSSSSS}",
    r"\toprule",
    r"{$\frac{\Delta a}{a}$} & & \multicolumn{3}{c}{\textbf{GT}} & \multicolumn{3}{c}{PIGP} \\",
    r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
    r"n & {$\sigma$} & {standard} & {extra\_n} & {extra\_n+$\sigma$} & "
    r"{standard} & {extra\_n} & {extra\_n+$\sigma$} \\",
    r"\midrule",
]
footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


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
    Group the three data-mode columns of one method into spans of *coinciding*
    data: consecutive columns whose raw per-run error arrays are identical are
    merged into a single (start, length) span.  This reproduces the template's
    merged cells (e.g. extra_npoints == extra_npoints_extra_sigma at sigma=0).
    Returns a list of (start_col, length) tuples covering columns 0..2.
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


rows = []
for n_idx, npoints in enumerate(n_list):
    for s_idx, (s_label, s_file) in enumerate(sigma_list):
        if s_idx == 0 and n_idx > 0:
            rows.append(r"\addlinespace")
        cell_mean = means[(n_idx, s_idx)]
        cell_raw = raw_errs[(n_idx, s_idx)]

        # merge spans per method (computed from the raw data)
        groups = {m: group_modes([cell_raw[m][d] for d in data_modes])
                  for m in methods}

        # per-method best (one representative value per rendered/merged cell)
        method_min = {}
        for m in methods:
            vals = [cell_mean[m][d] for d in data_modes]
            reps = [vals[start] for start, _ in groups[m] if not _is_nan(vals[start])]
            method_min[m] = min(reps) if reps else None
        valid = [v for v in method_min.values() if v is not None]
        overall_min = min(valid) if valid else None
        multi = len(data_modes) >= 2

        gt = render_region([cell_mean["GT"][d] for d in data_modes],
                           groups["GT"], method_min["GT"], overall_min, multi)
        pigp = render_region([cell_mean["PIGP"][d] for d in data_modes],
                             groups["PIGP"], method_min["PIGP"], overall_min, multi)

        rows.append(rf"{npoints} & {s_label} & {gt} & {pigp} \\")


# overall error per data mode: geometric mean of the cell means over all n, sigma
def _gmean(vals):
    vals = [v for v in vals if not _is_nan(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")


agg = {m: {d: _gmean([means[(ni, si)][m][d]
                      for ni in range(len(n_list)) for si in range(len(sigma_list))])
           for d in data_modes} for m in methods}
agg_mmin = {m: min([agg[m][d] for d in data_modes if not _is_nan(agg[m][d])], default=None)
            for m in methods}
_valid = [v for v in agg_mmin.values() if v is not None]
agg_omin = min(_valid) if _valid else None
_singletons = [(i, 1) for i in range(len(data_modes))]
gt_sum = render_region([agg["GT"][d] for d in data_modes], _singletons,
                       agg_mmin["GT"], agg_omin, len(data_modes) >= 2)
pigp_sum = render_region([agg["PIGP"][d] for d in data_modes], _singletons,
                         agg_mmin["PIGP"], agg_omin, len(data_modes) >= 2)
rows.append(r"\midrule")
rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{overall}}}} & {gt_sum} & {pigp_sum} \\")

table = "\n".join(header + rows + footer) + "\n"

out_path = os.path.join(base_dir, "Ex1_table.txt")
with open(out_path, "w") as f:
    f.write(table)

print(f"\nWrote table to {out_path}\n")
print(table)
