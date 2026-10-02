import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import Experiment1_Tripendulum.grid_evaluation.Tripendulum_GT as ex

import jax.numpy as jnp
from PCGP.jax import grid
import jax
from functools import partial

jax.config.update("jax_enable_x64", True)


if len(sys.argv) != 4:
    print("Usage: python script.py <npoints> <sigma> <mode>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
mode = str(sys.argv[3])

if mode == "known_u_known_sigma":
    folder = "known_u"
elif mode == "extra_npoints_extra_sigma":
    folder = "extra_npoints"
else:
    folder = "standard"

if sigma == 0:
    sigma = 0.001

data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", folder,
    "Ex1_GT_n%i_sigma%.3f_0.npz"%(npoints, sigma)
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

compiled_kernel = jax.jit(partial(
    ex.kernel,
    structure=structure
))


grids = jnp.meshgrid(l_vals, g_vals, ls_vals, A_vals, indexing="ij")
params = {"length": grids[0].flatten(), "g": grids[1].flatten(),  "lengthscale": grids[2].flatten(), "amplitude": grids[3].flatten()}
mll_values = grid.mll(params, train_x, train_y, sigma_arr, compiled_kernel)

output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data", mode,
    "GT_n%i_sigma%.3f_fulln%i.npz"%(npoints, sigma, n)
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



