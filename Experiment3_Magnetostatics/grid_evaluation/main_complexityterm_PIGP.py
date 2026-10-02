import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import Magnetostatics_PIGP as ex
import numpy as np
import jax.numpy as jnp


from PCGP.jax import grid
from prior import log_prior
import jax
import matplotlib.pyplot as plt
from functools import partial

jax.config.update("jax_enable_x64", True)
npoints = 3
num_tasks = 9

"""nn = 3
xx = jnp.linspace(0, 1, nn)
grids = jnp.meshgrid(xx, xx, xx+1, indexing = "ij")
train_x_x = jnp.concatenate([grids[0].flatten() for i in range(num_tasks)], axis = 0)
train_x_y = jnp.concatenate([grids[1].flatten() for i in range(num_tasks)], axis = 0)
train_x_z = jnp.concatenate([grids[2].flatten() for i in range(num_tasks)], axis = 0)
train_i = jnp.concatenate([jnp.ones_like(grids[0].flatten())*i for i in range(num_tasks)], axis = 0)
print(train_x_x.shape, train_x_y.shape, train_i.shape)
train_x = jnp.stack([train_x_x, train_x_y, train_x_z, train_i], axis = -1)"""
for sigma in [0.001, 0.01, 0.1, 0.2]:
    mode = "extra_npoints_extra_sigma"#"standard"#
    if mode == "extra_npoints_extra_sigma":
        data_path = os.path.join(os.path.dirname(__file__), "../input_data", "extra_npoints", f"Ex3_PIGP_n{npoints}_sigma{sigma:.3f}_0.npz")
    else:
        data_path = os.path.join(os.path.dirname(__file__), "../input_data", mode, f"Ex3_PIGP_n{npoints}_sigma{sigma:.3f}_0.npz")
    data = np.load(data_path)

    train_x = jnp.array(data["train_x"])
    test_x = jnp.array(data["test_x"])
    train_y = jnp.array(data["train_y"])
    test_y = jnp.array(data["test_y"])
    num_tasks = 9
    spacing = train_x[:,0][train_x[:,1]==0] #constraining lengthscale
    #print(spacing)
    min_len = (spacing[1]-spacing[0])
    max_len = (spacing[-1]-spacing[0])
    nu0_true = 1.
    structure = grid.build_structure(train_x, train_x, num_tasks=num_tasks)


    print(sigma, "sigma")


    n = 50
    nu0_vals = jnp.linspace(-1, 3, n)
    ls_vals = jnp.array([1.])#jnp.linspace(0.5, 1.5, 5)#
    A_vals = jnp.array([6.])#jnp.linspace(.1, 10., 5) #

    grids = jnp.meshgrid(nu0_vals, ls_vals, A_vals, indexing="ij")
    params = {"nu0": grids[0].flatten(), "lengthscale": grids[1].flatten(), "amplitude": grids[2].flatten()}

    sigma_array = jnp.ones(train_y.shape[0])*sigma
    sigma_array = sigma_array.at[train_x[..., -1] == 4].set(1e-3)
    sigma_array = sigma_array.at[train_x[..., -1] == 5].set(1e-3)
    compiled_kernel = jax.jit(partial(
        ex.kernel,
        structure=structure
    ))
    log_pri = log_prior(params, train_x,  compiled_kernel, sigma_array)

    print(log_pri)
    import scipy.special
    #log_prior = np.where(np.isfinite(log_prior), log_prior, -np.inf)
    #log_marginal = scipy.special.logsumexp(log_prior.reshape(20, 5, 5), axis=(-1,))
    #print(log_marginal)
    import matplotlib.pyplot as plt
    plt.plot(nu0_vals, log_pri)
    output_path = os.path.join(
        os.path.dirname(__file__),
        "output_data", mode,
        "PIGP_prior_sigma%.3f.npz"%sigma
    )
    np.savez_compressed(
                output_path,
                log_prior = log_pri,            
                nu0_vals = nu0_vals,
                ls_vals = ls_vals,
                A_vals = A_vals,
                param_order = np.array(["nu0_vals", "ls_vals", "A_vals"])
            )   

plt.show()
