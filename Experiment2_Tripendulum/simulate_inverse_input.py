import os
import torch
import numpy as np
import math
np.random.seed(123)


g_true = 9.81

def analytic_solution(train_x, BC, t1 = 5, wu = 1.6, g = 9.81, l = 2.5, sigma = 0, known_u = False):
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
    zeros = np.zeros_like(tt[index==4])
    if max(index) == 5: #PIGP
        y = np.concatenate([theta[:,0], theta[:,1], theta[:,2], u, u, u], axis=0)
    else: #GT
        y = np.concatenate([theta[:,0], theta[:,1], theta[:,2], u, zeros], axis=0)
    return y


BC = np.array([[3., 1., 0.08], [-0.2, -1, 0.1]]) #[[values at t=t0],[values at t = t1]]
t0 = 0
t1 = 5
sigma_list = [0.00, 0.01, 0.1, 0.2]
n_list = [3, 5, 10, 25]


modes = ["standard", "extra_npoints", "known_u"]


for sigma in sigma_list:
    for mode in modes:
        known_u = False
        if mode == "known_u":
            known_u = True
        for npoints in n_list:
            train_x_0 =  np.linspace(t0, t1, npoints)
            train_x_1 =  np.linspace(t0, t1, npoints)
            train_x_2 =  np.linspace(t0, t1, npoints)
            
            if mode == "standard":
                train_x_3 =  np.linspace(t0, t1, npoints)
                train_x_4 =  np.linspace(t0, t1, npoints) 
                
            elif mode == "extra_npoints":
                train_x_3 =  np.linspace(t0, t1, npoints)
                train_x_4 =  np.linspace(t0, t1, 25)#ghost tasking always uses 25 points for the control input, so we keep this fixed
            elif mode == "known_u":
                train_x_3 =  np.linspace(t0, t1, 25)
                train_x_4 =  np.linspace(t0, t1, 25)
            train_i = np.concatenate([np.zeros_like(train_x_0), np.ones_like(train_x_1), np.ones_like(train_x_2)*2, np.ones_like(train_x_3)*3, np.ones_like(train_x_4)*4], axis = 0)
            train_x = np.concatenate([train_x_0, train_x_1, train_x_2, train_x_3, train_x_4], axis = 0)
            full_train_x = np.stack([train_x, train_i], axis = -1)

            
            train_y = analytic_solution(full_train_x, BC,  t1 = t1, sigma = sigma, known_u = known_u)
            
            test_x_0 =  np.linspace(t0, t1, 201)
            test_i = np.concatenate([np.zeros_like(test_x_0), np.ones_like(test_x_0), np.ones_like(test_x_0)*2, np.ones_like(test_x_0)*3, np.ones_like(test_x_0)*4], axis = 0)
            test_x = np.concatenate([test_x_0]*5, axis = 0)
            full_test_x = np.stack([test_x, test_i], axis = -1)

            test_y = analytic_solution(full_test_x, BC, t1 = t1, sigma = 0)
            data_path = os.path.join(
                        os.path.dirname(__file__),
                        "input_data", mode,
                        "Ex2_n%i_sigma%.2f_GT.npz"%(npoints, sigma))
            np.savez_compressed(
                        data_path,
                        train_x = full_train_x,
                        test_x = full_test_x,
                        train_y = train_y,
                        test_y = test_y,
                    )
            print("GT saved", npoints, sigma)
#PIGP
for sigma in sigma_list:
    for mode in modes:
        known_u = False
        if mode == "known_u":
            known_u = True
        for npoints in n_list:
            train_x_0 =  np.linspace(t0, t1, npoints)
            train_x_1 =  np.linspace(t0, t1, npoints)
            train_x_2 =  np.linspace(t0, t1, npoints)
            if mode == "standard":
                train_x_3 =  np.linspace(t0, t1, npoints)
                train_x_4 =  np.linspace(t0, t1, npoints) 
                train_x_5 =  np.linspace(t0, t1, npoints) 
            elif mode == "extra_npoints":
                train_x_3 =  np.linspace(t0, t1, npoints)
                train_x_4 =  np.linspace(t0, t1, npoints)#ghost tasking always uses 25 points for the control input, so we keep this fixed
                train_x_5 =  np.linspace(t0, t1, npoints) 
            elif mode == "known_u":
                train_x_3 =  np.linspace(t0, t1, 25)
                train_x_4 =  np.linspace(t0, t1, 25)
                train_x_5 =  np.linspace(t0, t1, 25)
            train_i = np.concatenate([np.zeros_like(train_x_0), np.ones_like(train_x_1), np.ones_like(train_x_2)*2, np.ones_like(train_x_3)*3, np.ones_like(train_x_4)*4, np.ones_like(train_x_5)*5], axis = 0)
            train_x = np.concatenate([train_x_0, train_x_1, train_x_2, train_x_3, train_x_4, train_x_5], axis = 0)
            full_train_x = np.stack([train_x, train_i], axis = -1)

            train_y = analytic_solution(full_train_x, BC,  t1 = t1, sigma = sigma, known_u = known_u)
            
            test_x_0 =  np.linspace(t0, t1, 201)
        
            test_i = np.concatenate([np.zeros_like(test_x_0), np.ones_like(test_x_0), np.ones_like(test_x_0)*2, np.ones_like(test_x_0)*3, np.ones_like(test_x_0)*4, np.ones_like(test_x_0)*5], axis = 0)
            test_x = np.concatenate([test_x_0]*6, axis = 0)
        
            full_test_x = np.stack([test_x, test_i], axis = -1)

            test_y = analytic_solution(full_test_x, BC, t1 = t1, sigma = 0)
            data_path = os.path.join(
                        os.path.dirname(__file__),
                        "input_data", mode,
                        "Ex2_n%i_sigma%.2f_PIGP.npz"%(npoints, sigma))
            np.savez_compressed(
                        data_path,
                        train_x = full_train_x,
                        test_x = full_test_x,
                        train_y = train_y,
                        test_y = test_y,
                    )
            print("PIGP saved", npoints, sigma)
#import matplotlib.pyplot as plt
#for i in range(5):
#                plt.plot(train_x, train_y["GT"][:,i], 'o', label='Train GT')
#plt.show()