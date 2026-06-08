
import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve_triangular

import numpyro
import numpyro.distributions as dist 
import time
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
)
from numpyro.handlers import condition

                      

#@jax.jit
def kernel(x1, x2, parameters): #hab include noise rausgehaut
    N, M = x1.shape[0], x2.shape[0]
    xx = x1[:, :-1] 
    yy = x2[:, :-1]
    if xx.ndim == 1: #handle one dimensional input case
        xx = jnp.expand_dims(xx, axis=-1)
    if yy.ndim == 1:
        yy = jnp.expand_dims(yy, axis=-2) 
    xx = xx[:, jnp.newaxis, :] #for broadcasting: (N, 1, D)
    yy = yy[None, :, :]                      

    idx1 = x1[:, -1].astype(jnp.int32)
    idx2 = x2[:, -1].astype(jnp.int32)
    perm1 = jnp.argsort(idx1)
    perm2 = jnp.argsort(idx2)
    idx1_perm = idx1[perm1]
    idx2_perm = idx2[perm2]
    inv_idx1 = jnp.argsort(perm1)
    inv_idx2 = jnp.argsort(perm2)
    split_idx1 = jnp.cumsum(jnp.bincount(idx1_perm))[:-1]
    split_idx2 = jnp.cumsum(jnp.bincount(idx2_perm))[:-1]
    splits1 = jnp.array_split(xx[perm1], split_idx1, axis=0)
    splits2 = jnp.array_split(yy[:,perm2], split_idx2, axis=1)

    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    g = parameters['g']
    length = parameters['length']
    def k00(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/length**2
    def k01(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k02(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k03(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k04(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k05(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k10(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k11(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/length**2
    def k12(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k13(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k14(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k15(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k20(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k21(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k22(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/length**2
    def k23(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k24(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k25(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k30(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k31(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k32(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k33(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**4)
    def k34(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k35(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k40(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k41(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k42(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k43(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k44(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**4)
    def k45(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k50(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k51(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k52(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**2)
    def k53(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k54(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k55(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/(length**2*lengthscale**4)
    function_grid = [
        [k00, k01, k02, k03, k04, k05],
        [k10, k11, k12, k13, k14, k15],
        [k20, k21, k22, k23, k24, k25],
        [k30, k31, k32, k33, k34, k35],
        [k40, k41, k42, k43, k44, k45],
        [k50, k51, k52, k53, k54, k55],
    ]
                         
   
    rows = []
   
    for i, s1 in enumerate(splits1):
        row_blocks = []
        for j, s2 in enumerate(splits2):
            block = function_grid[i][j](s1, s2)
            row_blocks.append(block)
        rows.append(jnp.concatenate(row_blocks, axis=1))
    sorted_matrix = jnp.concatenate(rows, axis=0)
    K = jnp.empty((N, M), sorted_matrix.dtype)
    
    K = K.at[inv_idx1[:, None], inv_idx2[None, :]].set(sorted_matrix)   
    return K


def model(X, Y, sigma = 5e-2, priors = None): 
    num_tasks = 6
    #parameter sampling to be implemented later
    k = kernel(X, X, 0) #0 to be changed
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]*num_tasks), covariance_matrix=k),
        obs=Y)
                           
def gp_posterior_sample(key, kernel, X, Y, X_star, params, sigma):
    K = kernel(X, X, params) + sigma**2 * jnp.eye(X.shape[0])
    K_star = kernel(X, X_star, params)
    K_starstar = kernel(X_star, X_star, params)
    L = jnp.linalg.cholesky(K)
    
    z = solve_triangular(L, Y, lower=True)
    alpha = solve_triangular(L.T, z, lower=False)
    mu = K_star.T @ alpha
                           
    v = solve_triangular(L, K_star, lower=True)
    cov = K_starstar - v.T @ v
    
    # Sample
    L_post = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(cov.shape[0])) #to be seen how to handle jitter (hard coded?)
    eps = jax.random.normal(key, (cov.shape[0],))
    return mu + L_post @ eps
                           

def run_inference(model, rng_key, X, Y, num_samples, sigma, init_values, fixed_params = {}, priors = None): #maybe make init values a kwarg
    start = time.time()
    init_strategy = init_to_value(
            values=init_values) 
    model_with_opt_fixed_params = condition(model, data=fixed_params)
    kernel = NUTS(model_with_opt_fixed_params, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples= num_samples,
        num_chains=1,
        thinning=1,
        progress_bar=True, 
        )
    mcmc.run(rng_key, X, Y, sigma = sigma, priors = priors)
    mcmc.print_summary()
    print("MCMC elapsed time:", time.time() - start)
    return mcmc.get_samples() 

#bypasses numpyro and the model function and directly calls kernel. 
#@jax.jit
def single_mll(params, X, Y, sigma, order_of_parameters):
    # Formula: -0.5 * (y.T @ K_inv @ y + log|K| + N*log(2pi))
                           
    #expects params to be a vector of all parameters of the kernel in the order they are defined in the kernel function.
    p = {}
    for key in order_of_parameters:
        p[key] = params[order_of_parameters[key]]
    
    K = kernel(X, X, p)
    sigma_arr = jnp.asarray(sigma)

    if sigma_arr.ndim == 0:
        K += sigma**2 * jnp.eye(K.shape[0])
    else:
        K += sigma**2
    L = jnp.linalg.cholesky(K)
    L_inv_Y = solve_triangular(L, Y, lower=True) 
    fit_term = -0.5 * jnp.sum(L_inv_Y**2) 
    complexity_term = -jnp.sum(jnp.log(jnp.diagonal(L)))
    return fit_term + complexity_term 

mll = jax.vmap(single_mll, in_axes=(0, None, None, None, None))
                                 