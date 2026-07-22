import os
import numpy as np
import torch
from types import SimpleNamespace
from scipy.special import logsumexp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def load_run(data_dir, fname):
    """Load a results .npz file and return its arrays as torch tensors in a namespace."""
    raw = np.load(os.path.join(data_dir, fname), allow_pickle = True)
    ns = SimpleNamespace()
    for key in raw:
        try:
            setattr(ns, key, torch.tensor(raw[key]))
        except Exception:
            setattr(ns, key, raw[key])
    return ns


def plot_timeseries(fig, data, num_tasks, task_labels, outer_spec=None):
    """Plot stacked per-task time series into outer_spec, or the full figure if None."""
    if outer_spec is None:
        inner = gridspec.GridSpec(num_tasks, 1, figure=fig, hspace=0.05)
    else:
        inner = gridspec.GridSpecFromSubplotSpec(
            num_tasks, 1, subplot_spec=outer_spec, hspace=0.05
        )
    for i in range(num_tasks):
        mask_test  = data.test_x[..., -1] == i
        mask_train = data.train_x[..., -1] == i
        t_test  = data.test_x[mask_test, 0]
        t_train = data.train_x[mask_train, 0]

        ax = fig.add_subplot(inner[i, 0])
        ax.plot(t_test, data.mean[mask_test], "b", linewidth=2)
        ax.fill_between(t_test, data.lower[mask_test], data.upper[mask_test], alpha=0.3)
        ax.plot(t_test, data.test_y[mask_test], "r--")
        ax.plot(t_train, data.train_y[mask_train], "k*")
        ax.set_ylabel(task_labels[i], fontsize=22)

        if i < num_tasks - 1:
            ax.tick_params(labelbottom=False, bottom=False, labelsize=20)
        else:
            ax.set_xlabel(r"$t$", fontsize=22)
            ax.tick_params(labelsize=20)

        if i == 0:
            ax.set_title("Forward fit", fontsize=24)


_CANONICAL_ORDER = ["l_vals", "g_vals", "ls_vals", "A_vals"]


def _get_param_order(data):
    """Return the parameter axis order from data, falling back to canonical order."""
    if hasattr(data, "param_order"):
        return list(data.param_order)
    return [k for k in _CANONICAL_ORDER if hasattr(data, k)]


def _compute_marginal(mll, mode, param_order, n, marginalize_axes):
    """Marginalize a flat MLL grid: logsumexp for GT/PIGP, log-transform for Bayes.
    NaN entries (Cholesky failure for ill-conditioned params) are treated as -inf."""
    shape = (n,) * len(param_order)
    if mode == "Bayes":
        return -np.log(np.abs(mll.reshape(shape)))
    mll_clean = np.where(np.isfinite(mll), mll, -np.inf)
    return logsumexp(mll_clean.reshape(shape), axis=marginalize_axes)


def plot_grid_posterior(ax, data, mode, plot_axes=None, cmap="magenta", true_point=None):
    """Plot a single posterior panel, marginalizing over all axes not in plot_axes.

    plot_axes: list of 2 key names (e.g. ["l_vals", "g_vals"]), defaults to first two
               in param_order (physical parameters). Use ["ls_vals", "A_vals"] for the
               calibration posterior.
    """
    param_order = _get_param_order(data)
    if plot_axes is None:
        plot_axes = param_order[:2]

    x_key, y_key = plot_axes
    x_ax = param_order.index(x_key)
    y_ax = param_order.index(y_key)
    marginalize_axes = tuple(i for i in range(len(param_order)) if i not in (x_ax, y_ax))

    n = len(np.asarray(getattr(data, param_order[0])))
    mll = np.asarray(data.mll_values)
    marginal = _compute_marginal(mll, mode, param_order, n, marginalize_axes)

    # after marginalization the remaining axes are in sorted (param_order) order;
    # transpose if the user wants x and y swapped relative to that
    remaining = sorted([x_ax, y_ax])
    if remaining[0] != x_ax:
        marginal = marginal.T

    x_vals = np.asarray(getattr(data, x_key))
    y_vals = np.asarray(getattr(data, y_key))
    grids = np.meshgrid(x_vals, y_vals, indexing="ij")
    ax.pcolormesh(grids[0], grids[1], marginal, cmap=cmap, shading="auto")
    if true_point is not None:
        ax.plot(*true_point, color="C2", marker="o", markersize=5)


def plot_posterior_grid(path, mode, sigma_list, npoints_list, possible_n,
                        plot_axes=None, cmap="terrain", true_point=(2.5, 9.81),
                        title_suffix="marginalized over calibration"):
    """Plot a len(npoints_list) x len(sigma_list) grid of posterior panels for one mode."""
    n = possible_n[mode]
    fig, axes = plt.subplots(len(npoints_list), len(sigma_list),
                             figsize=(12, 12), constrained_layout=True)
    for i, npoints in enumerate(npoints_list):
        for j, sigma in enumerate(sigma_list):
            fname = f"{mode}_n{npoints}_sigma{sigma:.2f}_fulln{n}.npz"
            data = load_run(path, fname)
            plot_grid_posterior(axes[i, j], data, mode, plot_axes, cmap, true_point)
            axes[i, j].set_title(f"n={npoints} noise={sigma}")
    fig.suptitle(f"{mode} {title_suffix}")
    return fig, axes

