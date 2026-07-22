import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import Magnetostatics_GT as ex
import numpy as np
from PCGP.jax import grid
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial


if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma> ")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
if sigma == 0:
    sigma = 0.001
sigmaGT = 0.001
mode = "standard"

print(f"Threads used: {torch.get_num_threads()}")
if mode == "extra_npoints_extra_sigma":
    data_path = os.path.join(os.path.dirname(__file__), "../input_data", "extra_npoints", f"Ex3_GT_n{npoints}_sigma{sigma:.3f}.npz")
else:
    data_path = os.path.join(os.path.dirname(__file__), "../input_data", mode, f"Ex3_GT_n{npoints}_sigma{sigma:.3f}.npz")
data = np.load(data_path)

train_x = jnp.array(data["train_x"])
test_x = jnp.array(data["test_x"])
train_y = jnp.array(data["train_y"])
test_y = jnp.array(data["test_y"])
num_tasks = 8

spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
print(spacing)
min_len = (spacing[1]-spacing[0])
max_len = (spacing[-1]-spacing[0])
nu0_true = 1.

structure = grid.build_structure(train_x, train_x, num_tasks=num_tasks)





n = 20
nu0_vals = jnp.linspace(-1, 3, n)
ls_vals = jnp.array([1.])#jnp.linspace(min_len, max_len, 1)
A_vals = jnp.array([6.])#jnp.linspace(.1, 10, n) 

grids = jnp.meshgrid(nu0_vals, ls_vals, A_vals, indexing="ij")
params = {"nu0": grids[0].flatten(), "lengthscale": grids[1].flatten(), "amplitude": grids[2].flatten()}


compiled_kernel = jax.jit(partial(
    ex.kernel,
    structure=structure
))
sigma_array = jnp.ones(train_y.shape[0])*sigma
sigma_array = sigma_array.at[train_x[..., -1] == 2].set(1e-3)
mll_values = grid.mll(params, train_x, train_y, sigma_array, compiled_kernel)
#mll_reshaped = mll_values.reshape(n, n, n)
output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data", mode,
    "GT_n%i_sigma%.2f.npz"%(npoints, sigma)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,            
            nu0_vals = nu0_vals,
            ls_vals = ls_vals,
            A_vals = A_vals,
            param_order = np.array(["nu0_vals", "ls_vals", "A_vals"])
        )   



