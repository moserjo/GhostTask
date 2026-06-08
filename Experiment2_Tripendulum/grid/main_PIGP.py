import sys
import os
import torch
import Tripendulum_PIGP as ex
import numpy as np
import time
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)


if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])


data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", "known_u",
    "Ex2_n%i_sigma%.2f_PIGP.npz"%(npoints, sigma)
)
data = np.load(data_path)

train_x = jnp.array(data["train_x"])
test_x = jnp.array(data["test_x"])
train_y = jnp.array(data["train_y"])
test_y = jnp.array(data["test_y"])


num_tasks = max(test_x[:1])+1

spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
min_len = (spacing[1]-spacing[0])**2
max_len = (spacing[-1]-spacing[0])**2
length_true = 2.5
g_true = 9.81


sigma_arr = jnp.ones_like(train_y) * sigma**2

sigma_arr = sigma_arr.at[train_x[:, -1] == 4].set(1e-6)
sigma_arr = sigma_arr.at[train_x[:, -1] == 3].set(1e-6)
sigma_arr = sigma_arr.at[train_x[:, -1] == 5].set(1e-6)

n = 20
l_vals = jnp.linspace(0.5, 5., n)
g_vals = jnp.linspace(1, 15., n)
ls_vals = jnp.linspace(min_len, 10, n)
A_vals = jnp.linspace(.5, 100., n) 
order_of_parameters = {"length": 0, "g":1, "lengthscale": 2, "amplitude": 3} 
#L, G = jnp.meshgrid(l_vals, g_vals, indexing="ij")
#lengthscale = 2*jnp.ones_like(L)
#amplitude = 1.*jnp.ones_like(L)
if npoints == 25:
    results = []

    for i in range(n):
        A_chunk = A_vals[i:i+1]  # shape (1,)

        grids = jnp.meshgrid(l_vals, g_vals, ls_vals, A_chunk, indexing="ij")
        params_chunk = jnp.stack(grids, axis=-1).reshape(-1, 4)

        mll_chunk = ex.mll(params_chunk, train_x, train_y, sigma_arr, order_of_parameters)
        
        results.append(mll_chunk)

    mll_values = jnp.concatenate(results, axis=0)
    #mll_reshaped = mll_values.reshape(n, n, n, n)
else:
    

    grids = jnp.meshgrid(l_vals, g_vals, ls_vals, A_vals, indexing="ij")
    params_grid = jnp.stack(grids, axis=-1).reshape(-1, 4) #currently actually 30x30x30x30 grid
    


    mll_values = ex.mll(params_grid, train_x, train_y, sigma_arr, order_of_parameters)
    #mll_reshaped = mll_values.reshape(n, n, n, n)
if np.isnan(mll_values).any():
                    print("NaN values found in mll_reshaped")
output_path = os.path.join(
    os.path.dirname(__file__),
    "output","known_u_known_sigma",
    "PIGP_n%i_sigma%.2f_fulln%i.npz"%(npoints, sigma, n)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,            
            l_vals = l_vals,
            g_vals = g_vals,
            ls_vals = ls_vals,
            A_vals = A_vals,
        )   



