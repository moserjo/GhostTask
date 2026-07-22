import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import solve_triangular
jax.config.update("jax_enable_x64", True)
import os                           
import sys


if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])




def u_0(x):
    return np.zeros_like(x)
def v_0(x):
    return 3*np.sin(2*x)
def dv_0(x):
    return 3*2*np.cos(2*x)
a = 2.
def analytic_solution(full_train_x, u_0, v_0, dv_0, a):
    x = full_train_x[:,0]
    t = full_train_x[:,1]
    i = full_train_x[:,2]

    u = u_0(x[(i==0)]*np.exp(-t[(i==0)]))-a*(1-np.exp(-t[(i==0)]))*dv_0(x[(i==0)]*np.exp(-t[(i==0)]))
    
    v = v_0(x[(i==1)]*np.exp(-t[(i==1)]))
    y = jnp.concatenate([u, v], axis = 0)
    return y

#@jax.jit
def single_mll(a, X, Y, sigma):
    print(X.shape, X.shape)
    model_outputs = analytic_solution(X, u_0, v_0, dv_0, a)  
    squared_error = ((Y - model_outputs) ** 2).sum()
    log_likelihood = -squared_error/ (2 * sigma ** 2)
    return log_likelihood

mll = jax.vmap(single_mll, in_axes=(0, None, None, None))
   

n =1000
a_vals = jnp.linspace(-5,10, n)

order_of_parameters = {"a": 0} 


mode = "inverse_extra" 

if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma> ")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])

data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", "inverse",
    "Ex1_n%i_sigma%.2f_GT_extra_npoints.npz"%(npoints, sigma))
data = np.load(data_path)

train_x_GT = jnp.array(data["train_x"])
train_y = jnp.array(data["train_y"])
train_x = train_x_GT[train_x_GT[:,-1]<2] 
#print(train_x)
train_y = train_y[train_x_GT[:,-1]<2] 


test_x_GT = jnp.array(data["test_x"])
test_y = jnp.array(data["test_y"])
test_x = test_x_GT[test_x_GT[:,-1]<2] 
test_y = test_y[test_x_GT[:,-1]<2] 



mll_values = mll(a_vals, train_x, train_y, sigma)
mll_reshaped = mll_values

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
# Using origin='lower' is safer if your meshgrid was built with indexing='xy'
cp = plt.contourf(L, G, jnp.exp(mll_reshaped-jnp.max(mll_values)), levels=50, cmap = "rainbow")
plt.colorbar(cp, label='Log-Likelihood')


plt.figure()
max_idx = jnp.argmax(mll_reshaped)
idx = jnp.unravel_index(max_idx, mll_reshaped.shape)
best_l = l_vals[idx[0]]
best_g = g_vals[idx[1]]
plt.plot(test_x[:,0], analytic_solution(test_x, jnp.array([best_l, best_g]), npoints), label='Analytic Solution')
plt.plot(test_x[:,0], test_y, '--', label='Test Data')
plt.plot(train_x[:,0], train_y, 'k*', label='Train Data')

plt.show()"""


output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data",
    "Bayes_n%i_sigma%.2f_1d2.npz"%(npoints, sigma)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,            
            a_vals = a_vals,
        )   
