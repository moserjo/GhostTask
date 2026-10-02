import os
import torch
import numpy as np
import math
import itertools



g_true = 9.81

def analytic_solution(train_x, BC, t1 = 5, wu = 1.6, g = 9.81, l = 2.5, sigma = 0, known_u = False, sol_mode = "GT", seed = 1):
    np.random.seed(seed)
    tt = train_x[:,0]
    index = train_x[:,1]

    w = math.sqrt(g/l)
    A= 5.
    scaling = np.sin(t1*w)

    theta = np.zeros((tt[index == 0].shape[0], 3))
    for i in range(3):
        t = tt[index == i]
        theta[:,i] = np.sin(w*(t1-t))/scaling*BC[0,i] + np.sin(w*t)/scaling*BC[1,i] + (A/l)/(w**2 - wu**2)*(np.sin(wu*t) - np.sin(wu*t1)/scaling*np.sin(w*t)) 
    theta += sigma*np.random.randn(*theta.shape)
    t = tt[index == 3]
    u = A*np.sin(wu*t)
    if not known_u:
        u += np.random.randn(*u.shape)*sigma
    if sol_mode == "GT": #tasks 0-2 = theta, 3 = u, 4 = ghost task
        zeros = np.zeros_like(tt[index == 4])
        y = np.concatenate([theta[:,0], theta[:,1], theta[:,2], u, zeros], axis=0)
    else: #PIGP: tasks 0-2 = theta, 3-5 = u (one per equation)
        y = np.concatenate([theta[:,0], theta[:,1], theta[:,2], u, u, u], axis=0)
    return y


BC = np.array([[3., 1., 0.08], [-0.2, -1, 0.1]]) #[[values at t=t0],[values at t = t1]]
t0 = 0
t1 = 5
n_list = [3, 5, 10]


modes = ["standard", "extra_npoints", "known_u"]

for npoints in n_list:
    for mode in modes:
        known_u = (mode == "known_u")
        for run, sol_mode in itertools.product(range(10), ["GT", "PIGP"]):
            for s_idx, sigma in enumerate([0.2, 0.1, 0.01, 0.001]):
                # the seed depends on run and sigma only, NOT on sol_mode, so GT
                # and PIGP see the same noise realization while the runs differ
                seed = 1000*run + s_idx

                #ghost tasking always uses 25 points for the control input, so we keep this fixed
                n_u = 25 if known_u else npoints                                    #task 3 (control input)
                n_ghost = 25 if mode in ("extra_npoints", "known_u") else npoints   #task 4 (ghost task, GT only)
                t_obs = np.linspace(t0, t1, npoints)
                t_u = np.linspace(t0, t1, n_u)
                t_ghost = np.linspace(t0, t1, n_ghost)

                if sol_mode == "GT":
                    blocks = [t_obs, t_obs, t_obs, t_u, t_ghost]
                else:
                    blocks = [t_obs, t_obs, t_obs, t_u, t_u, t_u]
                train_x = np.concatenate(blocks, axis = 0)
                train_i = np.concatenate([np.ones_like(b)*k for k, b in enumerate(blocks)], axis = 0)
                full_train_x = np.stack([train_x, train_i], axis = -1)

                train_y = analytic_solution(full_train_x, BC, t1 = t1, sigma = sigma,
                                            known_u = known_u, sol_mode = sol_mode, seed = seed)

                test_x_0 = np.linspace(t0, t1, 201)
                n_tasks = 5 if sol_mode == "GT" else 6
                test_x = np.concatenate([test_x_0]*n_tasks, axis = 0)
                test_i = np.concatenate([np.ones_like(test_x_0)*k for k in range(n_tasks)], axis = 0)
                full_test_x = np.stack([test_x, test_i], axis = -1)

                test_y = analytic_solution(full_test_x, BC, t1 = t1, sigma = 0, sol_mode = sol_mode)

                input_dir = os.path.join(os.path.dirname(__file__), "input_data")
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
