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
if sigma == 0:
    sigma = 0.001




@jax.jit
def analytic_solution(train_x, p, npoints,
                      BC=jnp.array([[3., 1., 0.08], [-0.2, -1., 0.1]]),
                      t1=5.):
    l = p[0]
    g = p[1]
    wu = p[2]
    tt = train_x[:, 0]
    index = train_x[:, 1].astype(jnp.int32)

    w = jnp.sqrt(g / l)
    A = 5.
    scaling = jnp.sin(t1 * w)

    # Precompute all 3 theta components at once
    theta_all = (
        jnp.sin(w * (t1 - tt))[:, None] / scaling * BC[0] +
        jnp.sin(w * tt)[:, None] / scaling * BC[1] +
        (A / l) / (w**2 - wu**2) * (
            jnp.sin(wu * tt)[:, None] -
            jnp.sin(wu * t1) / scaling * jnp.sin(w * tt)[:, None]
        )
    )  # shape (N, 3)

    u = A * jnp.sin(wu * tt)  # shape (N,)

    # Select correct output per index
    y = jnp.where(
        index[:, None] < 3,
        theta_all,
        jnp.zeros_like(theta_all)
    )
    y = jnp.sum(y * jax.nn.one_hot(index, 4)[:, :3], axis=1)
    u_mask = (index == 3)
    y = y + u * u_mask
    return y

@jax.jit
def single_mll(params, X, Y, sigma, npoints):
  
    model_outputs = analytic_solution(X, params, npoints)  #wu = 6.4 makes more possibilities
    squared_error = ((Y - model_outputs) ** 2).sum()
    log_likelihood = -squared_error/ (2 * sigma ** 2)
    return log_likelihood

mll = jax.jit(jax.vmap(single_mll, in_axes=(0, None, None, None,  None))
)    

n =50
l_vals = jnp.linspace(0.5, 5., n)
g_vals = jnp.linspace(1, 15., n)
wu_vals = jnp.linspace(0.5, 8, n)
L, G, WU = jnp.meshgrid(l_vals, g_vals, wu_vals, indexing="ij")

params_grid = jnp.stack([L, G, WU], axis=-1).reshape(-1, 3)
order_of_parameters = {"length": 0, "g":1, "WU":2} 



data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data", "standard",
    "Ex1_GT_n%i_sigma%.3f_0.npz"%(npoints, sigma)
)
data = np.load(data_path)

train_x_GT = jnp.array(data["train_x"])
train_y = jnp.array(data["train_y"])
train_x = train_x_GT[train_x_GT[:,1]<4] 
train_y = train_y[train_x_GT[:,1]<4] 


test_x_GT = jnp.array(data["test_x"])
test_y = jnp.array(data["test_y"])
test_x = test_x_GT[test_x_GT[:,1]<4] 
test_y = test_y[test_x_GT[:,1]<4] 



mll_values = mll(params_grid, train_x, train_y, sigma,  npoints)
mll_reshaped = mll_values.reshape(n, n, n)


output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data", "Bayes",
    "Bayes_n%i_sigma%.3f_fulln%i.npz"%(npoints, sigma, n)
)
np.savez_compressed(
            output_path,
            mll_values = mll_values,
            l_vals = l_vals,
            g_vals = g_vals,
            wu_vals = wu_vals,
            param_order = np.array(["l_vals", "g_vals", "wu_vals"]),
        )
