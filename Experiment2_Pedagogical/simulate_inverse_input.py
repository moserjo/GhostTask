import os
import numpy as np
import math
import torch
np.random.seed(1)
torch.set_default_dtype(torch.float64)
import itertools                                    # new

def u_0(x):
    return np.zeros_like(x)
def v_0(x):
    return 3*np.sin(2*x)
def dv_0(x):
    return 3*2*np.cos(2*x)
a = 2.
def analytic_solution(full_train_x, u_0, v_0, dv_0, a, noise = 0, sol_mode = "GT"):
    x = full_train_x[:,0]
    t = full_train_x[:,1]
    i = full_train_x[:,2]

    u = u_0(x[(i==0)]*np.exp(-t[(i==0)]))-a*(1-np.exp(-t[(i==0)]))*dv_0(x[(i==0)]*np.exp(-t[(i==0)]))
    if noise:
        u += np.random.randn(u.shape[0],)*noise
    v = v_0(x[(i==1)]*np.exp(-t[(i==1)]))
    if noise:
        v += np.random.randn(v.shape[0])*noise
    if sol_mode == "GT":
        GT = np.zeros_like(t[(i==2)])
        y = np.concatenate([u, v, GT], axis = 0)
    else:
        GT = np.zeros_like(t[(i==2)])
        y = np.concatenate([u, v, GT, GT], axis = 0)
    return y

n_test = 51
for npoints in [3, 5, 10]:
    for mode in ["extra_npoints", "standard"]:
        for run, sol_mode in itertools.product(range(10), ["GT", "PIGP"]):   # was: for sol_mode in [...]
            for s_idx, sigma in enumerate([0.2, 0.1, 0.01, 0.001]):          # was: for noise in [...]
                # the seed depends on run and sigma only, NOT on sol_mode, so GT
                # and PIGP see the same noise realization while the runs differ
                np.random.seed(1000*run + s_idx)
                
                if mode == "extra_npoints":
                    xx = torch.linspace(0, 1, npoints)
                    tt = torch.linspace(0, 1, npoints)
                    xGT = torch.linspace(0, 1, 15)
                    tGT = torch.linspace(0, 1, 15)
                    train_x_GT = torch.cartesian_prod(xGT, tGT)
                    train_i_GT = torch.ones_like(train_x_GT[:,0])*2
                    train_i_GT2 = torch.ones_like(train_x_GT[:,0])*3
                    train_x_0 = torch.cartesian_prod(xx, tt)
                    train_i_0 = torch.zeros_like(train_x_0[:,0])
                    train_x_1 = torch.cartesian_prod(xx, tt)
                    train_i_1 = torch.ones_like(train_x_1[:,0])
                    if sol_mode == "GT":
                        train_x = torch.cat([train_x_0, train_x_1, train_x_GT], dim = 0)
                        train_i = torch.cat([train_i_0, train_i_1, train_i_GT], dim = 0)
                    else:
                        train_x = torch.cat([train_x_0, train_x_1, train_x_GT, train_x_GT], dim = 0)
                        train_i = torch.cat([train_i_0, train_i_1, train_i_GT, train_i_GT2], dim = 0)
                    full_train_x = torch.stack([train_x[:,0], train_x[:,1], train_i], dim = -1).numpy()
                elif mode == "standard":
                    xx = torch.linspace(0, 1, npoints)
                    tt = torch.linspace(0, 1, npoints)
                    train_x_GT = torch.cartesian_prod(xx, tt)
                    train_i_GT = torch.ones_like(train_x_GT[:,0])*2
                    train_x_0 = train_x_GT#[train_x_GT[:,1] == 0]
                    train_i_0 = torch.zeros_like(train_x_0[:,0])
                    train_x_1 = train_x_GT#[train_x_GT[:,1] == 0]
                    train_i_1 = torch.ones_like(train_x_1[:,0])
                    if sol_mode == "GT":
                        train_x = torch.cat([train_x_0, train_x_1, train_x_GT], dim = 0)
                        train_i = torch.cat([train_i_0, train_i_1, train_i_GT], dim = 0)
                    else:
                        train_x = torch.cat([train_x_0, train_x_1, train_x_GT, train_x_GT], dim = 0)
                        train_i = torch.cat([train_i_0, train_i_1, train_i_GT, train_i_1*3], dim = 0)
                    full_train_x = torch.stack([train_x[:,0], train_x[:,1], train_i], dim = -1).numpy()
           
                train_y = analytic_solution(full_train_x, u_0, v_0, dv_0, a,  noise = sigma, sol_mode=sol_mode)

                xx = np.linspace(0, 1, n_test)
                tt = np.linspace(0, 1, n_test)
                x, t = np.meshgrid(xx, tt)
                xx = torch.linspace(0, 1, n_test)
                tt = torch.linspace(0, 1, n_test)

                test_x = torch.cartesian_prod(xx, tt)
                test_0 = torch.zeros_like(test_x[:,0])
                test_1 = torch.ones_like(test_x[:,0])
                test_2 = torch.ones_like(test_x[:,0])*2

                if sol_mode == "GT":
                    test_x = torch.cat([test_x, test_x, test_x], dim = 0)
                    test_i = torch.cat([test_0, test_1, test_2], dim = 0)
                else:
                    test_x = torch.cat([test_x, test_x, test_x, test_x], dim = 0)
                    test_i = torch.cat([test_0, test_1, test_2, test_1*3], dim = 0)
                full_test_x = torch.stack([test_x[:,0], test_x[:,1], test_i], dim = -1).numpy()


                test_y = analytic_solution(full_test_x, u_0, v_0, dv_0, a, sol_mode=sol_mode)            
                input_dir = os.path.join(os.path.dirname(__file__), "input_data")
                os.makedirs(input_dir, exist_ok=True)
                mode_dir = os.path.join(input_dir, mode)
                os.makedirs(mode_dir, exist_ok=True)
                data_path = os.path.join(mode_dir,
                                        "Ex1_%s_n%i_sigma%.3f_%i.npz"%(sol_mode, npoints, sigma, run))
                np.savez_compressed(
                                        data_path,
                                        train_x = full_train_x,
                                        test_x = full_test_x,
                                        train_y = train_y,
                                        test_y = test_y,
                                    )
                print(sol_mode, "saved: npoints=%i, sigma=%.3f, run %i" % (npoints, sigma, run))
