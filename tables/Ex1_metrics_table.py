"""
Ex1_metrics_table.py
--------------------
Builds the LaTeX results table for Experiment 1 (Pedagogical), including the
projective Ghost-Tasking variant.

For every combination of
    npoints   in {3, 5, 10}
    sigma     in {0.001, 0.01, 0.1, 0.2}  (file label uses %.3f -> 0.001/.../0.200)
    method    in {GT, GT_projective, PIGP}
    data_mode in {standard, extra_npoints_extra_sigma}
it reads the first 10 runs (run = 0..9), computes the relative error of the
learned parameter a (true value 2):

        rel_err = |a - 2| / 2

then takes the MEDIAN over the runs.

In each row (one per npoints/sigma) the lowest median across the 6
method/data-mode cells is marked bold, the best cell within one method italic.

Output: Ex1_table.txt  (copy-paste ready LaTeX).
"""

import os
import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
base_dir = os.path.dirname(os.path.abspath(__file__))       # .../tables
repo_dir = os.path.dirname(base_dir)                        # repository root
output_data_ex1 = os.path.join(
    repo_dir, "Experiment1_Pedagogical", "optimisation", "Ex1output"
)

runs = 10                                   # first 10 runs: 0 .. runs-1
true_a = 2.0

n_list = [3, 5, 10]                          # table rows (npoints)
# (display label for the row, sigma value used in the filename via %.3f)
sigma_list = [("0.001", 0.001),
              ("0.01", 0.010),
              ("0.1", 0.100),
              ("0.2", 0.200)]

methods = ["GT", "GT_projective", "PIGP"]    # column groups
data_modes = ["standard", "extra_npoints_extra_sigma"]

# column-group / data-mode headers
METHOD_DISPLAY = {
    "GT": r"\textbf{GT}",
    "GT_projective": r"\textbf{GT projective}",
    "PIGP": r"PIGP",
}
MODE_DISPLAY = {
    "standard": "standard",
    "extra_npoints_extra_sigma": r"extra\_n+$\sigma$",
}

NUM_FMT = "{:.2g}"                           # 2 significant figures


# --------------------------------------------------------------------------- #
# data loading + metric
# --------------------------------------------------------------------------- #
def relative_errors(npoints, sigma_file, method, data_mode):
    """Relative errors |a-2|/2 of the learned parameter for the first `runs` runs."""
    data_dir = os.path.join(output_data_ex1, data_mode)
    errs = []
    for run in range(runs):
        fname = f"Ex1_n{npoints}_sigma{sigma_file:.3f}_{method}_{run}.npz"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] missing file: {os.path.relpath(path, repo_dir)}")
            continue
        raw = np.load(path, allow_pickle=True)
        a = float(np.array(raw["learned_parameters"]).reshape(-1)[-1])
        errs.append(abs(a - true_a) / true_a)
    return np.asarray(errs, dtype=float)


def median(values):
    """Median over the runs; NaN if the cell has no data."""
    if values.size == 0:
        return np.nan
    return float(np.median(values))


# --------------------------------------------------------------------------- #
# compute, for every cell, both the raw per-run error array (used to detect
# coinciding data modes -> merged cells) and the median over the runs.
#   raw_errs[(n_idx, s_idx)][method][data_mode]  -> np.ndarray
#   medians[(n_idx, s_idx)][method][data_mode]   -> float
# --------------------------------------------------------------------------- #
raw_errs = {}
medians = {}

for n_idx, npoints in enumerate(n_list):
    for s_idx, (s_label, s_file) in enumerate(sigma_list):
        cell_raw = {m: {} for m in methods}
        cell_median = {m: {} for m in methods}
        for method in methods:
            for dmode in data_modes:
                errs = relative_errors(npoints, s_file, method, dmode)
                cell_raw[method][dmode] = errs
                cell_median[method][dmode] = median(errs)
        raw_errs[(n_idx, s_idx)] = cell_raw
        medians[(n_idx, s_idx)] = cell_median


# --------------------------------------------------------------------------- #
# build the LaTeX table
# --------------------------------------------------------------------------- #
n_modes = len(data_modes)
n_cols = len(methods) * n_modes

# \multicolumn group headers + matching \cmidrule ranges
group_cells = []
cmidrules = []
for g_idx, method in enumerate(methods):
    lo = 3 + g_idx * n_modes
    hi = lo + n_modes - 1
    group_cells.append(rf"\multicolumn{{{n_modes}}}{{c}}{{{METHOD_DISPLAY[method]}}}")
    cmidrules.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
mode_cells = [f"{{{MODE_DISPLAY[d]}}}" for _ in methods for d in data_modes]

header = [
    r"% Requires \usepackage{booktabs} and \usepackage{siunitx} in the preamble.",
    r"\begin{table}[]",
    # mode=match is what makes \itshape actually show: by default siunitx sets
    # numbers in math mode, where digits have no italic shape, so detect-shape
    # silently does nothing and the per-method-best cells look unmarked.
    r"\sisetup{mode=match, round-mode=figures, round-precision=2, scientific-notation=true, "
    r"table-format=1.2e-1, detect-weight=true, detect-shape=true, detect-family=true}",
    rf"\begin{{tabular}}{{ll{'S' * n_cols}}}",
    r"\toprule",
    r"{$\frac{\Delta a}{a}$} & & " + " & ".join(group_cells) + r" \\",
    " ".join(cmidrules),
    r"n & {$\sigma$} & " + " & ".join(mode_cells) + r" \\",
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
    Group the data-mode columns of one method into spans of *coinciding* data:
    consecutive columns whose raw per-run error arrays are identical are merged
    into a single (start, length) span.
    Returns a list of (start_col, length) tuples covering all data modes.
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
        cell_median = medians[(n_idx, s_idx)]
        cell_raw = raw_errs[(n_idx, s_idx)]

        # merge spans per method (computed from the raw data)
        groups = {m: group_modes([cell_raw[m][d] for d in data_modes])
                  for m in methods}

        # per-method best (one representative value per rendered/merged cell)
        method_min = {}
        for m in methods:
            vals = [cell_median[m][d] for d in data_modes]
            reps = [vals[start] for start, _ in groups[m] if not _is_nan(vals[start])]
            method_min[m] = min(reps) if reps else None
        valid = [v for v in method_min.values() if v is not None]
        overall_min = min(valid) if valid else None
        multi = n_modes >= 2

        regions = [render_region([cell_median[m][d] for d in data_modes],
                                 groups[m], method_min[m], overall_min, multi)
                   for m in methods]

        rows.append(rf"{npoints} & {s_label} & " + " & ".join(regions) + r" \\")


# overall error per data mode: geometric mean of the cell medians over all n, sigma
def _gmean(vals):
    vals = [v for v in vals if not _is_nan(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")


agg = {m: {d: _gmean([medians[(ni, si)][m][d]
                      for ni in range(len(n_list)) for si in range(len(sigma_list))])
           for d in data_modes} for m in methods}
agg_mmin = {m: min([agg[m][d] for d in data_modes if not _is_nan(agg[m][d])], default=None)
            for m in methods}
_valid = [v for v in agg_mmin.values() if v is not None]
agg_omin = min(_valid) if _valid else None
_singletons = [(i, 1) for i in range(n_modes)]
sum_regions = [render_region([agg[m][d] for d in data_modes], _singletons,
                             agg_mmin[m], agg_omin, n_modes >= 2)
               for m in methods]
rows.append(r"\midrule")
rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{overall}}}} & "
            + " & ".join(sum_regions) + r" \\")

table = "\n".join(header + rows + footer) + "\n"

out_path = os.path.join(base_dir, "Ex1_table.txt")
with open(out_path, "w") as f:
    f.write(table)

print(f"\nWrote table to {out_path}\n")
print(table)
