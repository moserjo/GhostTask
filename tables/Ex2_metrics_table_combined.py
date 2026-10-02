"""
Ex2_metrics_table_combined.py
-----------------------------
Builds THE Experiment 2 (Tripendulum) LaTeX results table: one single table
that merges the former "standard" and "known u" tables.

All three inferred quantities are packed into every cell, written as

        param0 / param1 / param2      e.g.   0.1/0.2/0.01

with
    param 0 :  |Delta l / l|          = |(l - l_true) / l_true|
    param 1 :  |Delta g / g|          = |(g - g_true) / g_true|
    param 2 :  |Delta (g/l)/(g/l)|    = |((g/l) - (g_true/l_true)) / (g_true/l_true)|

Column hierarchy (three header levels):

    measured u                                   | known u
    GT                       | PIGP              | GT            | PIGP
    standard | extra_n+sigma | standard          | known_sigma   | known_sigma

  * Under `measured u` only GT is split by data mode: PIGP's runs for
    `extra_npoints(_extra_sigma)` are bit-identical to its `standard` runs (the
    extra-point modes leave PIGP's training inputs unchanged), so PIGP gets a
    single `standard` column.
  * Under `known u` only `known_u_known_sigma` is compared, for both methods.

Every cell reports the MEDIAN over the runs.  Highlighting is applied PER
PARAMETER inside each combined cell, so the three slash-separated numbers are
styled independently:
    bold    -> best cell WITHIN ITS REGIME (measured u / known u).  The regimes
               are not comparable (known u is a far easier problem and would
               otherwise win every bold), so best-of is taken per regime.
    italic  -> best cell within one method block that spans >1 data mode
               (i.e. only the measured-u GT block).
Adjacent data-mode cells of one method whose raw per-run data coincide are
merged with \\multicolumn.

Output: Ex2_table_combined.txt
"""

import os
import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
base_dir = os.path.dirname(os.path.abspath(__file__))       # .../tables
repo_dir = os.path.dirname(base_dir)                        # repository root
output_data_ex2 = os.path.join(
    repo_dir, "Experiment2_Tripendulum", "optimisation", "output_data"
)

runs = 10                                   # first 10 runs: 0 .. runs-1

n_list = [3, 5, 10]                          # table rows (npoints)
# (display label for the row, sigma value used in the filename via %.3f)
sigma_list = [("0.001", 0.001),
              ("0.01", 0.010),
              ("0.1", 0.100),
              ("0.2", 0.200)]

# the table's column blocks, in display order:
#   (regime key, method, [data modes shown for this block])
BLOCKS = [
    ("measured", "GT",   ["standard", "extra_npoints_extra_sigma"]),
    ("measured", "PIGP", ["standard"]),
    ("known",    "GT",   ["known_u_known_sigma"]),
    ("known",    "PIGP", ["known_u_known_sigma"]),
]

# the three parameters packed into every cell, in display order
PARAMS = (0, 1, 2)

# display names for the three header levels
REGIME_DISPLAY = {
    "measured": r"measured $u$",
    "known": r"known $u$",
}
METHOD_DISPLAY = {
    "GT": r"\textbf{GT}",
    "PIGP": r"PIGP",
}
MODE_DISPLAY = {
    "standard": "standard",
    "extra_npoints": r"extra\_n",
    "extra_npoints_extra_sigma": r"extra\_n+$\sigma$",
    "known_u": r"known\_u",
    "known_u_known_sigma": r"known\_$\sigma$",
}

# LaTeX label of each analysed parameter (used in the corner cell)
PARAM_LABEL = {
    0: r"\frac{\Delta \ell}{\ell}",
    1: r"\frac{\Delta g}{g}",
    2: r"\frac{\Delta g/\ell}{g/\ell}",
}

NUM_FMT = "{:.2g}"                           # 2 significant figures


# --------------------------------------------------------------------------- #
# data loading + metric
# --------------------------------------------------------------------------- #
def _rel_error(learned, true, param_idx):
    """Relative error of the requested parameter (0:l, 1:g, 2:ratio g/l)."""
    if param_idx == 0:
        return abs((learned[0] - true[0]) / true[0])
    if param_idx == 1:
        return abs((learned[1] - true[1]) / true[1])
    learned_ratio = learned[1] / learned[0]
    true_ratio = true[1] / true[0]
    return abs((learned_ratio - true_ratio) / true_ratio)


def relative_errors(npoints, sigma_file, method, data_mode, param_idx):
    """Per-run relative errors of `param_idx` for the first `runs` runs."""
    data_dir = os.path.join(output_data_ex2, data_mode)
    errs = []
    for run in range(runs):
        fname = f"Ex2_{method}_n{npoints}_sigma{sigma_file:.3f}_{run}.npz"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] missing file: {os.path.relpath(path, repo_dir)}")
            continue
        raw = np.load(path, allow_pickle=True)
        learned = np.array(raw["learned_parameters"]).reshape(-1)
        true = np.array(raw["true_parameters"]).reshape(-1)
        errs.append(float(_rel_error(learned, true, param_idx)))
    return np.asarray(errs, dtype=float)


def median(values):
    """Median over the runs; NaN if the cell has no data."""
    if values.size == 0:
        return np.nan
    return float(np.median(values))


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


def combined_cell_tex(param_vals, block_mins, regime_mins, block_multi):
    r"""
    Render one cell holding all three parameters as `p0/p1/p2`, each number
    typeset with \num{} (siunitx handles rounding + scientific notation).  Each
    parameter is styled independently:
      best in the regime -> bold, best in the block -> italic (blocks with >1
      data mode only).
    """
    parts = []
    for p in PARAMS:
        value = param_vals[p]
        if _is_nan(value):
            parts.append("-")
            continue
        body = r"\num{" + f"{value:.6g}" + "}"
        if regime_mins[p] is not None and value == regime_mins[p]:
            body = r"{\bfseries " + body + "}"
        elif block_multi and block_mins[p] is not None and value == block_mins[p]:
            body = r"{\itshape " + body + "}"
        parts.append(body)
    return "/".join(parts)


def _modes_equal(a, b):
    """True if two modes coincide across ALL three parameters' raw run data."""
    return all(a[p].size > 0 and np.array_equal(a[p], b[p]) for p in PARAMS)


def group_modes(arrays_by_mode):
    """
    Group a block's data-mode columns into spans of *coinciding* data:
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


def render_region(mode_medians, groups, block_mins, regime_mins, block_multi):
    """
    Render one block's column region as LaTeX, honouring merged spans (booktabs,
    no vertical rules).  `mode_medians` is a list over modes of {param_idx: median}.
    """
    tokens = []
    for start, length in groups:
        vals = mode_medians[start]               # coinciding -> any member works
        s = combined_cell_tex(vals, block_mins, regime_mins, block_multi)
        if length > 1:                            # merged span -> centered
            tokens.append(rf"\multicolumn{{{length}}}{{c}}{{{s}}}")
        else:
            tokens.append(s)
    return " & ".join(tokens)


def _block_extrema(mode_medians, groups):
    """
    For one block in one row: per-parameter minimum over the (merged) cells.
    Returns {param_idx: min_value_or_None}.
    """
    out = {}
    for p in PARAMS:
        reps = [mode_medians[start][p] for start, _ in groups
                if not _is_nan(mode_medians[start][p])]
        out[p] = min(reps) if reps else None
    return out


def _regime_extrema(block_mins):
    """Per-parameter minimum over all blocks of one regime."""
    out = {}
    for p in PARAMS:
        vals = [bm[p] for bm in block_mins if bm[p] is not None]
        out[p] = min(vals) if vals else None
    return out


def _spans(labels_and_widths, first_col):
    """
    Render one header level as \\multicolumn cells plus the matching
    \\cmidrule ranges.  `labels_and_widths` is a list of (label, width).
    """
    cells, rules = [], []
    col = first_col
    for label, width in labels_and_widths:
        if width > 1:
            cells.append(rf"\multicolumn{{{width}}}{{c}}{{{label}}}")
        else:
            cells.append(label)
        rules.append(rf"\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width
    return " & ".join(cells), " ".join(rules)


# --------------------------------------------------------------------------- #
# compute every cell: raw per-run arrays + median over the runs
#   raw_errs[(n_idx, s_idx)][block_idx][data_mode][param] -> np.ndarray
#   medians[(n_idx, s_idx)][block_idx][data_mode][param]  -> float
# --------------------------------------------------------------------------- #
raw_errs = {}
medians = {}

for n_idx, npoints in enumerate(n_list):
    for s_idx, (s_label, s_file) in enumerate(sigma_list):
        cell_raw = []
        cell_median = []
        for _regime, method, modes in BLOCKS:
            raw_b, median_b = {}, {}
            for dmode in modes:
                raw_p, median_p = {}, {}
                for p in PARAMS:
                    errs = relative_errors(npoints, s_file, method, dmode, p)
                    raw_p[p] = errs
                    median_p[p] = median(errs)
                raw_b[dmode] = raw_p
                median_b[dmode] = median_p
            cell_raw.append(raw_b)
            cell_median.append(median_b)
        raw_errs[(n_idx, s_idx)] = cell_raw
        medians[(n_idx, s_idx)] = cell_median


# --------------------------------------------------------------------------- #
# build the LaTeX table
# --------------------------------------------------------------------------- #
block_widths = [len(modes) for _r, _m, modes in BLOCKS]
n_cols = sum(block_widths)

# level 1: regimes (contiguous blocks sharing the same regime key)
regime_spans = []
for regime, _m, modes in BLOCKS:
    if regime_spans and regime_spans[-1][0] == regime:
        regime_spans[-1][1] += len(modes)
    else:
        regime_spans.append([regime, len(modes)])
lvl1_cells, lvl1_rules = _spans(
    [(REGIME_DISPLAY[r], w) for r, w in regime_spans], first_col=3)

# level 2: methods
lvl2_cells, lvl2_rules = _spans(
    [(METHOD_DISPLAY[m], len(modes)) for _r, m, modes in BLOCKS], first_col=3)

# level 3: data modes
lvl3_cells = " & ".join(MODE_DISPLAY[d] for _r, _m, modes in BLOCKS for d in modes)

corner = "$" + "/".join(PARAM_LABEL[p] for p in PARAMS) + "$"
spec = "ll" + "c" * n_cols

# three numbers per cell make this table very wide -> shrink to \textwidth
header = [
    r"% Requires \usepackage{booktabs}, \usepackage{siunitx} and "
    r"\usepackage{graphicx} (for \resizebox) in the preamble.",
    r"\begin{table}[]",
    # mode=match is what makes \itshape actually show: by default siunitx sets
    # numbers in math mode, where digits have no italic shape, so detect-shape
    # silently does nothing and the per-block-best cells look unmarked.
    r"\sisetup{mode=match, round-mode=figures, round-precision=2, scientific-notation=true, "
    r"detect-weight=true, detect-shape=true, detect-family=true}",
    r"\resizebox{\textwidth}{!}{%",
    rf"\begin{{tabular}}{{{spec}}}",
    r"\toprule",
    rf"{corner} & & {lvl1_cells} \\",
    lvl1_rules,
    rf" & & {lvl2_cells} \\",
    lvl2_rules,
    rf"n & $\sigma$ & {lvl3_cells} \\",
    r"\midrule",
]
caption = (r"Relative errors (each cell: $\frac{\Delta \ell}{\ell}$ / "
           r"$\frac{\Delta g}{g}$ / $\frac{\Delta g/\ell}{g/\ell}$); "
           r"bold marks the best cell within its regime "
           r"(measured $u$ / known $u$).")
footer = [r"\bottomrule", r"\end{tabular}%", r"}",
          rf"\caption{{{caption}}}", r"\end{table}"]

rows = []
for n_idx, npoints in enumerate(n_list):
    for s_idx, (s_label, s_file) in enumerate(sigma_list):
        if s_idx == 0 and n_idx > 0:
            rows.append(r"\addlinespace")
        cell_median = medians[(n_idx, s_idx)]
        cell_raw = raw_errs[(n_idx, s_idx)]

        groups = [group_modes([cell_raw[b][d] for d in BLOCKS[b][2]])
                  for b in range(len(BLOCKS))]
        block_min = [_block_extrema([cell_median[b][d] for d in BLOCKS[b][2]],
                                    groups[b]) for b in range(len(BLOCKS))]
        # best per parameter within each regime (blocks are not comparable
        # across regimes: known u is a far easier problem)
        regime_min = {r: _regime_extrema([block_min[b] for b in range(len(BLOCKS))
                                          if BLOCKS[b][0] == r])
                      for r, _w in regime_spans}

        regions = [render_region([cell_median[b][d] for d in BLOCKS[b][2]],
                                 groups[b], block_min[b],
                                 regime_min[BLOCKS[b][0]], len(BLOCKS[b][2]) >= 2)
                   for b in range(len(BLOCKS))]

        rows.append(rf"{npoints} & {s_label} & " + " & ".join(regions) + r" \\")


# overall error per column / parameter: geometric mean of the cell medians
# over all n, sigma
def _gmean(vals):
    vals = [v for v in vals if not _is_nan(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")


agg = [{d: {p: _gmean([medians[(ni, si)][b][d][p]
                       for ni in range(len(n_list))
                       for si in range(len(sigma_list))])
            for p in PARAMS}
        for d in BLOCKS[b][2]} for b in range(len(BLOCKS))]
agg_groups = [[(i, 1) for i in range(len(BLOCKS[b][2]))] for b in range(len(BLOCKS))]
agg_bmin = [_block_extrema([agg[b][d] for d in BLOCKS[b][2]], agg_groups[b])
            for b in range(len(BLOCKS))]
agg_rmin = {r: _regime_extrema([agg_bmin[b] for b in range(len(BLOCKS))
                                if BLOCKS[b][0] == r])
            for r, _w in regime_spans}
sum_regions = [render_region([agg[b][d] for d in BLOCKS[b][2]], agg_groups[b],
                             agg_bmin[b], agg_rmin[BLOCKS[b][0]],
                             len(BLOCKS[b][2]) >= 2)
               for b in range(len(BLOCKS))]
rows.append(r"\midrule")
rows.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{overall}}}} & "
            + " & ".join(sum_regions) + r" \\")

table = "\n".join(header + rows + footer) + "\n"

out_path = os.path.join(base_dir, "Ex2_table_combined.txt")
with open(out_path, "w") as f:
    f.write(table)

print(f"\nWrote table to {out_path}\n")
print(table)
