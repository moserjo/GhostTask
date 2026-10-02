import os
import torch
import numpy as np
import math
np.random.seed(123)


g_true = 9.81
A = 5.  # forcing amplitude, u(t) = A*sin(wu*t)


def bc_to_ic(BC, t1=5, wu=1.6, g=9.81, l=2.5):
    """Convert boundary conditions BC = [[theta(t0)], [theta(t1)]] (with t0 = 0)
    into initial conditions IC = [[theta(t0)], [theta'(t0)]] for the forced
    pendulum theta'' + w^2 theta = (A/l) sin(wu t)."""
    w = math.sqrt(g/l)
    scaling = np.sin(t1*w)
    P = (A/l)/(w**2 - wu**2)

    theta0 = np.asarray(BC[0], dtype=float)
    theta1 = np.asarray(BC[1], dtype=float)
    dtheta0 = (-w*np.cos(w*t1)/scaling*theta0
               + w/scaling*theta1
               + P*(wu - w*np.sin(wu*t1)/scaling))
    return np.stack([theta0, dtheta0], axis=0)


def analytic_solution(train_x, IC, t1 = 5, wu = 1.6, g = 9.81, l = 2.5, sigma = 0, known_u = False):
    # IC = [[theta(t0)], [theta'(t0)]] with t0 = 0
    tt = train_x[:,0]
    index = train_x[:,1]

    w = math.sqrt(g/l)
    P = (A/l)/(w**2 - wu**2)

    thetas = []
    derivatives = []
    for i in range(3):
        t = tt[index == i]
        if max(index) == 8:
            dt = tt[index == 6+i]
        else:
            dt = tt[index == 5+i]
        C1 = IC[0, i]
        C2 = (IC[1, i] - P*wu)/w
        th = C1*np.cos(w*t) + C2*np.sin(w*t) + P*np.sin(wu*t)
        th += sigma*np.random.randn(*th.shape)
        dth = -C1*w*np.sin(w*dt) + w*C2*np.cos(w*dt) + wu*P*np.cos(wu*dt)
        dth += sigma*np.random.randn(*dth.shape)
        thetas.append(th)
        derivatives.append(dth)

    t = tt[index == 3]
    u = A*np.sin(wu*t)
    if not known_u:
        u += np.random.randn(*u.shape)*sigma
    zeros = np.zeros_like(tt[index==4])
    
    if max(index) == 8: #PIGP
        y = np.concatenate(thetas+[u, u, u]+derivatives, axis=0)
    else: #GT
        y = np.concatenate(thetas +[u, zeros]+derivatives, axis=0)
    return y


BC = np.array([[3., 1., 0.08], [-0.2, -1, 0.1]]) #[[values at t=t0],[values at t = t1]]
t0 = 0
t1 = 5

# Initial conditions IC = [[theta(t0)], [theta'(t0)]] equivalent to the chosen BC.
IC = bc_to_ic(BC, t1=t1)
print("IC (theta(t0), theta'(t0)):\n", IC)
sigma_list = [0.00, 0.01, 0.1, 0.2]
n_list = [3, 5, 10, 25]



npoints = 10
sigma = 0.0
train_x_0 =  np.array([0])
train_x_1 =  np.array([0])
train_x_2 =  np.array([0])

train_x_3 =  np.linspace(t0, t1, npoints)
train_x_4 =  np.linspace(t0, t1, npoints) 

train_x_5 =  np.array([0])
train_x_6 =  np.array([0])
train_x_7 =  np.array([0])

train_i = np.concatenate([np.zeros_like(train_x_0), 
                          np.ones_like(train_x_1), 
                          np.ones_like(train_x_2)*2, 
                          np.ones_like(train_x_3)*3, 
                          np.ones_like(train_x_4)*4,
                          np.ones_like(train_x_5)*5,
                          np.ones_like(train_x_6)*6,
                          np.ones_like(train_x_7)*7,
                          ], axis = 0)
train_x = np.concatenate([train_x_0, train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6, train_x_7], axis = 0)
full_train_x = np.stack([train_x, train_i], axis = -1)


train_y = analytic_solution(full_train_x, IC,  t1 = t1, sigma = sigma)

test_x_0 =  np.linspace(t0, t1, 201)
test_i = np.concatenate([np.zeros_like(test_x_0), 
                         np.ones_like(test_x_0), 
                         np.ones_like(test_x_0)*2, 
                         np.ones_like(test_x_0)*3, 
                         np.ones_like(test_x_0)*4,
                         np.ones_like(test_x_0)*5,
                         np.ones_like(test_x_0)*6,
                         np.ones_like(test_x_0)*7
                         ], axis = 0)
test_x = np.concatenate([test_x_0]*8, axis = 0)
full_test_x = np.stack([test_x, test_i], axis = -1)

test_y = analytic_solution(full_test_x, IC, t1 = t1, sigma = 0)
data_path = os.path.join(
            os.path.dirname(__file__),
            "input_data", "forward",
            "Ex1_GT.npz")
np.savez_compressed(
            data_path,
            train_x = full_train_x,
            test_x = full_test_x,
            train_y = train_y,
            test_y = test_y,
        )
print("GT saved", npoints, sigma)


#GT+
train_x_0 =  np.array([0])
train_x_1 =  np.array([0])
train_x_2 =  np.array([0])

train_x_3 =  np.linspace(t0, t1, npoints)
train_x_4 =  np.linspace(t0, t1, 20) 

train_x_5 =  np.array([0])
train_x_6 =  np.array([0])
train_x_7 =  np.array([0])

train_i = np.concatenate([np.zeros_like(train_x_0), 
                          np.ones_like(train_x_1), 
                          np.ones_like(train_x_2)*2, 
                          np.ones_like(train_x_3)*3, 
                          np.ones_like(train_x_4)*4,
                          np.ones_like(train_x_5)*5,
                          np.ones_like(train_x_6)*6,
                          np.ones_like(train_x_7)*7,
                          ], axis = 0)
train_x = np.concatenate([train_x_0, train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6, train_x_7], axis = 0)
full_train_x = np.stack([train_x, train_i], axis = -1)


train_y = analytic_solution(full_train_x, IC,  t1 = t1, sigma = sigma)

test_x_0 =  np.linspace(t0, t1, 201)
test_i = np.concatenate([np.zeros_like(test_x_0), 
                         np.ones_like(test_x_0), 
                         np.ones_like(test_x_0)*2, 
                         np.ones_like(test_x_0)*3, 
                         np.ones_like(test_x_0)*4,
                         np.ones_like(test_x_0)*5,
                         np.ones_like(test_x_0)*6,
                         np.ones_like(test_x_0)*7
                         ], axis = 0)
test_x = np.concatenate([test_x_0]*8, axis = 0)
full_test_x = np.stack([test_x, test_i], axis = -1)

test_y = analytic_solution(full_test_x, IC, t1 = t1, sigma = 0)
data_path = os.path.join(
            os.path.dirname(__file__),
            "input_data", "forward",
            "Ex1_GT+.npz")
np.savez_compressed(
            data_path,
            train_x = full_train_x,
            test_x = full_test_x,
            train_y = train_y,
            test_y = test_y,
        )
print("GT+ saved", npoints, sigma)
#PIGP

train_x_0 =  np.array([0.])
train_x_1 =  np.array([0.,])#np.linspace(t0, t1, npoints)#np.array([t0, t1])
train_x_2 =  np.array([0.,])#np.linspace(t0, t1, npoints)#np.array([t0, t1])

train_x_3 =  np.linspace(t0, t1, npoints)
train_x_4 =  np.linspace(t0, t1, npoints) 
train_x_5 =  np.linspace(t0, t1, npoints)

train_x_6 =  np.array([0.])
train_x_7 =  np.array([0.])
train_x_8 =  np.array([0.])

train_i = np.concatenate([np.zeros_like(train_x_0), 
                          np.ones_like(train_x_1), 
                          np.ones_like(train_x_2)*2, 
                          np.ones_like(train_x_3)*3, 
                          np.ones_like(train_x_4)*4, 
                          np.ones_like(train_x_5)*5,
                          np.ones_like(train_x_6)*6,
                          np.ones_like(train_x_7)*7,
                          np.ones_like(train_x_8)*8,
                          ], axis = 0)
train_x = np.concatenate([train_x_0, 
                          train_x_1, 
                          train_x_2, 
                          train_x_3, 
                          train_x_4, 
                          train_x_5,
                          train_x_6, 
                          train_x_7, 
                          train_x_8, 
                          ], axis = 0)
full_train_x = np.stack([train_x, train_i], axis = -1)
train_y = analytic_solution(full_train_x, IC,  t1 = t1, sigma = sigma)

test_x_0 =  np.linspace(t0, t1, 201)
test_i = np.concatenate([np.zeros_like(test_x_0), 
                         np.ones_like(test_x_0), 
                         np.ones_like(test_x_0)*2, 
                         np.ones_like(test_x_0)*3, 
                         np.ones_like(test_x_0)*4,
                         np.ones_like(test_x_0)*5,
                         np.ones_like(test_x_0)*6,
                         np.ones_like(test_x_0)*7,
                         np.ones_like(test_x_0)*8,
                         ], axis = 0)
test_x = np.concatenate([test_x_0]*9, axis = 0)
full_test_x = np.stack([test_x, test_i], axis = -1)




test_y = analytic_solution(full_test_x, IC, t1 = t1, sigma = 0)
data_path = os.path.join(
            os.path.dirname(__file__),
            "input_data", "forward",
            "Ex1_PIGP.npz")
np.savez_compressed(
            data_path,
            train_x = full_train_x,
            test_x = full_test_x,
            train_y = train_y,
            test_y = test_y,
        )
print("PIGP saved", npoints, sigma)