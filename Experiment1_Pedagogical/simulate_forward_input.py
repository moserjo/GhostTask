import os
import numpy as np
import math
import torch
np.random.seed(1)
torch.set_default_dtype(torch.float64)

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
mode = "yesextrapoint" #or "yesextrapoint"
for mode in ["yesextrapoint", "noextrapoint"]:
    for sol_mode in ["GT", "PIGP"]:
        for n_train in [10, 15]:
            noise = 0
            if mode == "yesextrapoint":
                xx = torch.linspace(-1/n_train, 1, n_train+1)
                tt = torch.linspace(0., 1, n_train)
            if mode == "noextrapoint":
                xx = torch.linspace(0, 1, n_train)
                tt = torch.linspace(0, 1, n_train)
            train_x_GT = torch.cartesian_prod(xx, tt)
            train_i_GT = torch.ones_like(train_x_GT[:,0])*2
            train_x_0 = train_x_GT[train_x_GT[:,1] == 0]
            train_i_0 = torch.zeros_like(train_x_0[:,0])
            train_x_1 = train_x_GT[train_x_GT[:,1] == 0]
            train_i_1 = torch.ones_like(train_x_1[:,0])
            if sol_mode == "GT":
                train_x = torch.cat([train_x_0, train_x_1, train_x_GT], dim = 0)
                train_i = torch.cat([train_i_0, train_i_1, train_i_GT], dim = 0)
            else:
                train_x = torch.cat([train_x_0, train_x_1, train_x_GT, train_x_GT], dim = 0)
                train_i = torch.cat([train_i_0, train_i_1, train_i_GT, torch.ones_like(train_x_GT[:,0])*3], dim = 0)
            full_train_x = torch.stack([train_x[:,0], train_x[:,1], train_i], dim = -1).numpy()
        
            train_y = analytic_solution(full_train_x, u_0, v_0, dv_0, a,  noise = noise, sol_mode=sol_mode)

            xx = torch.linspace(-1/n_train, 1, n_test)
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
            print("test_y" , test_y.shape)
                    
            data_path = os.path.join(
                                    os.path.dirname(__file__),
                                    "input_data", "forward", 
                                    "Ex1_n%i_%s_%s.npz"%(n_train, sol_mode, mode))
            np.savez_compressed(
                                    data_path,
                                    train_x = full_train_x,
                                    test_x = full_test_x,
                                    train_y = train_y,
                                    test_y = test_y,
                                )
            print("saved")

print(full_test_x)