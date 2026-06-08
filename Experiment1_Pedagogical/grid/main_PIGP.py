import sys
import os
import Pedagogical_PIGP as ex
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

#add jax precision and maybe key


mode = "inverse_extra" 

if len(sys.argv) != 3:
    print("Usage: python script.py <npoints> <sigma> ")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])

data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data",
    "Ex1_n%i_sigma%.2f_PIGP_%s.npz"%(npoints, sigma, mode))

data = np.load(data_path)
train_x = jnp.array(data["train_x"])
print(train_x.shape)
test_x = jnp.array(data["test_x"])
train_y = jnp.array(data["train_y"])
test_y = jnp.array(data["test_y"])

num_tasks = max(test_x[:1])+1

spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
min_len = (spacing[1]-spacing[0])**2
max_len = (spacing[-1]-spacing[0])**2
a_true = 2.






n = 100
a_vals = jnp.linspace(1,20, n)
ls_vals = jnp.array([.1,  5, 10.]) #jnp.linspace(0.1, 5)
A_vals = jnp.array([0.1, 10., 100]) #jnp.linspace(1, 1., 5) 
order_of_parameters = {"a": 0, "lengthscale": 2, "amplitude": 3} 


grids = jnp.meshgrid(a_vals, ls_vals, A_vals, indexing="ij")
params_grid = jnp.stack(grids, axis=-1).reshape(-1, 3) 


mll_values = ex.mll(params_grid, train_x, train_y, sigma, order_of_parameters)
#mll_reshaped = mll_values.reshape(n, n, n)
output_path = os.path.join(
    os.path.dirname(__file__),
    "output",
    "PIGP_n%i_sigma%.2f_3x3x100_2.npz"%(npoints, sigma)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,            
            a_vals = a_vals,
            ls_vals = ls_vals,
            A_vals = A_vals,
        )   



