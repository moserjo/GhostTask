import sys
import os
import Tripendulum_GT as ex
import numpy as np
import jax.numpy as jnp
from PCGP.jax import grid
import jax
import matplotlib.pyplot as plt
from functools import partial

jax.config.update("jax_enable_x64", True)


if len(sys.argv) != 4:
    print("Usage: python script.py <npoints> <sigma> <mode>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
mode = str(sys.argv[3])

if mode == "known_u_known_sigma" or mode == "known_u":
    folder = "known_u"
elif mode == "extra_npoints" or mode ==  "extra_npoints_extra_sigma":
    folder = "extra_npoints"
else:
    folder = "standard"
data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", folder,
    "Ex2_n%i_sigma%.2f_GT.npz"%(npoints, sigma)
)
data = np.load(data_path)
num_tasks = 5

train_x = jnp.array(data["train_x"])
test_x = jnp.array(data["test_x"])
train_y = jnp.array(data["train_y"])
test_y = jnp.array(data["test_y"])
structure = grid.build_structure(train_x, train_x, num_tasks=num_tasks)



spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
min_len = (spacing[1]-spacing[0])
max_len = (spacing[-1]-spacing[0])
length_true = 2.5
g_true = 9.81

if sigma == 0:
    sigma = 0.001
sigma_arr = jnp.ones_like(train_y) * sigma
if mode == "extra_npoints_extra_sigma":
    sigma_arr = sigma_arr.at[train_x[:, -1] == 4].set(1e-3)
if mode == "known_u_known_sigma":
    sigma_arr = sigma_arr.at[train_x[:, -1] == 3].set(1e-3)

n = 20
l_vals = jnp.linspace(0.5, 5., n)
g_vals = jnp.linspace(1, 15., n)
ls_vals = jnp.linspace(min_len, max_len, n)
A_vals = jnp.linspace(.5, 100., n) 
#order_of_parameters = {"length": 0, "g":1, "lengthscale": 2, "amplitude": 3} 

compiled_kernel = jax.jit(partial(
    ex.kernel,
    structure=structure
))

if npoints == 25:
    results = []
    for i in range(n):
        A_chunk = A_vals[i:i+1]  # shape (1,)
        grids = jnp.meshgrid(l_vals, g_vals, ls_vals, A_chunk, indexing="ij")
        params_chunk = {"length": grids[0].flatten(), "g": grids[1].flatten(), "lengthscale": grids[2].flatten(), "amplitude": grids[3].flatten()}
        mll_chunk = grid.mll(params_chunk, train_x, train_y, sigma_arr, compiled_kernel)
        results.append(mll_chunk.reshape(n, n, n, 1))  # (l, g, ls, A_i)
    mll_values = jnp.concatenate(results, axis=3).reshape(-1)  # (l, g, ls, A) → flat
else:
    grids = jnp.meshgrid(l_vals, g_vals, ls_vals, A_vals, indexing="ij")
    params = {"length": grids[0].flatten(), "g": grids[1].flatten(),  "lengthscale": grids[2].flatten(), "amplitude": grids[3].flatten()}
    mll_values = grid.mll(params, train_x, train_y, sigma_arr, compiled_kernel)

output_path = os.path.join(
    os.path.dirname(__file__),
    "output", mode,
    "GT_n%i_sigma%.2f_fulln%i.npz"%(npoints, sigma, n)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,
            l_vals = l_vals,
            g_vals = g_vals,
            ls_vals = ls_vals,
            A_vals = A_vals,
            param_order = np.array(["l_vals", "g_vals", "ls_vals", "A_vals"]),
        )   



