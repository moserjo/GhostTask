"""
ockhams_razor_figure.py
-----------------------
Illustrates WHY adding zero-task (ghost) collocation points, and lowering their
noise, biases the inferred parameter for PIGP but not for GT.

For a Gaussian process the negative log marginal likelihood is
    -mll = 1/2 y^T (K+Sigma)^-1 y  +  1/2 log|K+Sigma|  +  const.
The parameter estimate balances two forces:
    * data-fit curvature   d^2/dθ^2 [1/2 y^T M^-1 y]   (Fisher information on θ)
    * Occam pull           d/dθ     [1/2 log|M|]        (log-det / complexity term)
and the resulting bias ~ Occam pull / data-fit curvature.

Two sweeps, evaluated at the true parameter:
  (A) n_extra : number of points on the ZERO task(s)  (extra_npoints knob)
  (B) sigma_n : observation noise on the ZERO task(s) (extra_sigma knob)
for  GT-Ex2 (param g: in observed AND ghost tasks)  vs
     PIGP-Ex1 (param a: only in ghost tasks).

Derivatives via central finite differences (float64).
Output: graphs/ockhams_razor_mechanism.png
"""
import sys, os
import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.abspath(__file__))
SIGMA2 = 0.1**2          # observation-noise floor on the DATA tasks


def load_kernel_module(reldir, modname):
    d = os.path.join(REPO, reldir)
    for p in (d, os.path.join(d, "..", ".."), os.path.join(d, "..")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return __import__(modname)


def make_model(ex, train_x, train_y, modified_parameters):
    lik = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=torch.full((train_y.shape[0],), SIGMA2))
    return ex.PCGP_Model(train_x, train_y, lik, modified_parameters)


def subset(tx, ty, zero_tasks, k):
    """All observed-task rows + first k rows of each zero task. Returns x, y, tasks."""
    tasks = tx[:, -1].round().long()
    keep = []
    for t in torch.unique(tasks).tolist():
        rows = torch.where(tasks == t)[0]
        if t in zero_tasks:
            rows = rows[:k]
        keep.append(rows)
    keep = torch.cat(keep)
    return tx[keep], ty[keep], tasks[keep]


def terms(model, pname, theta, tx_sub, noise_vec, ty_sub):
    """(0.5 log|M|, 0.5 y^T M^-1 y) at parameter value theta with diagonal noise_vec."""
    model.covar_module._set_param(pname, float(theta))
    with torch.no_grad():
        K = model.covar_module(tx_sub).to_dense()
        M = K + torch.diag(noise_vec)
        M = 0.5 * (M + M.T)
        logdet = 0.5 * torch.linalg.slogdet(M)[1]
        fit = 0.5 * ty_sub @ torch.linalg.solve(M, ty_sub)
    return float(logdet), float(fit)


def forces(model, pname, p0, txk, noise_vec, tyk):
    """(|Occam pull|, |data-fit curvature|) at the true parameter p0."""
    h = 0.01 * p0
    ldp, ftp = terms(model, pname, p0 + h, txk, noise_vec, tyk)
    ldm, ftm = terms(model, pname, p0 - h, txk, noise_vec, tyk)
    ld0, ft0 = terms(model, pname, p0,     txk, noise_vec, tyk)
    occam = abs((ldp - ldm) / (2 * h))
    curv = abs((ftp - 2 * ft0 + ftm) / h ** 2)
    return occam, curv


def make(cfg):
    ex = load_kernel_module(cfg["reldir"], cfg["modname"])
    data = np.load(cfg["data"], allow_pickle=True)
    tx = torch.tensor(data["train_x"]); ty = torch.tensor(data["train_y"])
    model = make_model(ex, tx, ty, cfg["modified_parameters"])
    for name, val in cfg["fixed"].items():
        model.covar_module._set_param(name, val)
    return model, tx, ty


def sweep_nextra(cfg):
    model, tx, ty = make(cfg)
    ns, occ, cur = [], [], []
    for k in cfg["ks"]:
        txk, tyk, _ = subset(tx, ty, cfg["zero_tasks"], k)
        noise = torch.full((txk.shape[0],), SIGMA2)         # uniform soft noise
        o, c = forces(model, cfg["pname"], cfg["ptrue"], txk, noise, tyk)
        ns.append(k * len(cfg["zero_tasks"])); occ.append(o); cur.append(c)
    return np.array(ns), np.array(occ), np.array(cur)


def sweep_sigma(cfg, sig2_list):
    """Fix n_extra at the max; vary the ZERO-task noise sigma_n^2."""
    model, tx, ty = make(cfg)
    txk, tyk, tasks = subset(tx, ty, cfg["zero_tasks"], cfg["ks"][-1])
    is_zero = torch.tensor([int(t) in cfg["zero_tasks"] for t in tasks])
    occ, cur = [], []
    for s2 in sig2_list:
        noise = torch.full((txk.shape[0],), SIGMA2)
        noise[is_zero] = s2                                 # extra_sigma knob
        o, c = forces(model, cfg["pname"], cfg["ptrue"], txk, noise, tyk)
        occ.append(o); cur.append(c)
    return np.array(sig2_list), np.array(occ), np.array(cur)


GT = dict(
    reldir="Experiment2_Tripendulum/optimisation/GT", modname="Tripendulum_GT",
    data=os.path.join(REPO, "Experiment2_Tripendulum/output_data/extra_npoints/Ex2_GT_n5_sigma0.10_0.npz"),
    modified_parameters={"length": [2.5, False, gpytorch.constraints.Interval(0.5, 4.)],
                         "g": [9.81, False, gpytorch.constraints.Positive()],
                         "amplitude": [31.42, False, gpytorch.constraints.Positive()],
                         "lengthscale": [1.27, False, gpytorch.constraints.Positive()]},
    fixed={"length": 2.5, "amplitude": 31.42, "lengthscale": 1.27},
    pname="g", ptrue=9.81, zero_tasks=[4], ks=list(range(1, 26)),
    label="GT-Ex2  (g in observed + ghost)", color="C0")

PIGP = dict(
    reldir="Experiment1_Pedagogical/optimisation/PIGP", modname="Experiment1_Pedagogical_PIGP",
    data=os.path.join(REPO, "Experiment1_Pedagogical/optimisation/output_data/extra_npoints/Ex1_n3_sigma0.10_PIGP_0.npz"),
    modified_parameters={"a": [2.0, False, False],
                         "lengthscale": [0.6443, False, gpytorch.constraints.Positive()],
                         "amplitude": [9.6608, False, gpytorch.constraints.Positive()]},
    fixed={"amplitude": 9.6608, "lengthscale": 0.6443},
    pname="a", ptrue=2.0, zero_tasks=[2, 3],
    ks=[1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 180, 225],
    label="PIGP-Ex1  (a only in ghost)", color="C3")

PIGP3 = dict(
    reldir="Experiment3_Magnetostatics/optimization/PIGP", modname="Experiment3_PIGP",
    data=os.path.join(REPO, "Experiment3_Magnetostatics/output_data/standard/Ex3_PIGP_n5_sigma0.100_0.npz"),
    modified_parameters={"nu0": [1.0, False, False],
                         "amplitude": [9.0294, False, gpytorch.constraints.Positive()],
                         "lengthscale": [0.3713, False, gpytorch.constraints.Positive()]},
    fixed={"amplitude": 9.0294, "lengthscale": 0.3713},
    pname="nu0", ptrue=1.0, zero_tasks=[4, 5],          # Jy, Jz are identically zero
    ks=[1, 2, 3, 5, 8, 12, 20, 35, 55, 85, 125],
    label="PIGP-Ex3  (nu0; Jy,Jz genuinely zero)", color="C1")

SIG2 = np.logspace(0, -8, 17)     # sigma_n^2 from 1 down to 1e-8 (extra_sigma sweep)

def panel(a, x, occ, cur, xlabel, title, invert=False, logx=False):
    a.plot(x, cur, "o-", color="C0", label="data-fit curvature (Fisher info)")
    a.plot(x, occ, "s--", color="C3", label="|Occam pull| (log-det)")
    a.set_yscale("log")
    if logx:
        a.set_xscale("log")
    if invert:
        a.invert_xaxis()
    a.set_xlabel(xlabel); a.set_title(title, fontsize=11)
    a.legend(fontsize=8); a.grid(True, which="both", ls=":", alpha=.4)


# 3 rows {GT-Ex2, PIGP-Ex1, PIGP-Ex3} x 2 cols {vary n_extra, vary sigma_n}
fig, ax = plt.subplots(3, 2, figsize=(13, 14))
for row, cfg in enumerate([GT, PIGP, PIGP3]):
    print(f"{cfg['label']}: n_extra sweep ...")
    n, o, c = sweep_nextra(cfg)
    print(f"{cfg['label']}: sigma_n sweep ...")
    s, so, sc = sweep_sigma(cfg, SIG2)
    panel(ax[row, 0], n, o, c, r"$n_\mathrm{extra}$ (zero-task points)",
          cfg["label"] + r"  -- vary $n_\mathrm{extra}$")
    panel(ax[row, 1], s, so, sc, r"$\sigma_n^2$  (smaller $\to$ extra_sigma)",
          cfg["label"] + r"  -- vary $\sigma_n$", invert=True, logx=True)
    ax[row, 1].axvline(SIGMA2, color="k", ls=":", alpha=.6)
    print(f"   bias(n_extra) {o[0]/c[0]:.2e}->{o[-1]/c[-1]:.2e} | "
          f"bias(sigma_n) {so[0]/sc[0]:.2e}->{so[-1]/sc[-1]:.2e}")

fig.suptitle("Occam-razor forces on the inferred parameter: curvature (data anchor) vs Occam pull\n"
             "GT-Ex2 anchored (curvature wins) | PIGP-Ex1 semi-anchored | "
             "PIGP-Ex3 orphaned: Occam OVERTAKES curvature -> bias",
             fontsize=12)
fig.tight_layout()
out = os.path.join(REPO, "graphs", "ockhams_razor_mechanism.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
