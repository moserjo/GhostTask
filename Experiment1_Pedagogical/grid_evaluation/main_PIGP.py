import sys
import os
import Pedagogical_PIGP as ex
from PCGP.jax import grid
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from functools import partial

#add jax precision and maybe key


mode = "inverse_extra" 

if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma> ")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])

data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", "inverse",
    "Ex1_n%i_sigma%.2f_PIGP_extra_npoints.npz"%(npoints, sigma))

data = np.load(data_path)
train_x = jnp.array(data["train_x"])
test_x = jnp.array(data["test_x"])
train_y = jnp.array(data["train_y"])
test_y = jnp.array(data["test_y"])
num_tasks = 4

spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
min_len = (spacing[1]-spacing[0])
max_len = (spacing[-1]-spacing[0])
a_true = 2.

structure = grid.build_structure(train_x, train_x, num_tasks=num_tasks)





n = 200
a_vals = jnp.linspace(1, 3, n)
ls_vals = jnp.linspace(min_len, max_len, 5)
A_vals = jnp.linspace(.1, 10, 5) #jnp.array([0.1, 10., 100]) #
#order_of_parameters = {"a": 0, "lengthscale": 2, "amplitude": 3} 


grids = jnp.meshgrid(a_vals, ls_vals, A_vals, indexing="ij")
params = {"a": grids[0].flatten(), "lengthscale": grids[1].flatten(), "amplitude": grids[2].flatten()}


compiled_kernel = jax.jit(partial(
    ex.kernel,
    structure=structure
))

sigma_array = jnp.ones(train_y.shape[0])*sigma
sigma_array = sigma_array.at[train_x[:, -1] == 2].set(1e-3)
sigma_array = sigma_array.at[train_x[:, -1] == 3].set(1e-3)
mll_values = grid.mll(params, train_x, train_y, sigma_array, compiled_kernel)
#mll_reshaped = mll_values.reshape(n, n, n)
output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data", "yesextraGTsigma_inverse_extra",
        "PIGP_n%i_sigma%.2f.npz"%(npoints, sigma)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,            
            a_vals = a_vals,
            ls_vals = ls_vals,
            A_vals = A_vals,
            param_order = np.array(["a_vals", "ls_vals", "A_vals"])
        )   



