
import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve_triangular
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist 
import time
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
)
from numpyro.handlers import condition

                      

@jax.jit
def kernel(x1, x2, parameters): #hab include noise rausgehaut
   

    # 1. Separate features and task indices
    # Shape of xx: (N, 1), Shape of yy: (1, M)
    xx = x1[:, :-1] 
    yy = x2[:, :-1]
    if xx.ndim == 1: #handle one dimensional input case
        xx = jnp.expand_dims(xx, axis=-1)
    if yy.ndim == 1:
        yy = jnp.expand_dims(yy, axis=-2) 
    xx = xx[:, jnp.newaxis, :] #for broadcasting: (N, 1, D)
    yy = yy[jnp.newaxis, :, :]                      
                           
    idx1 = x1[:, -1].astype(jnp.int32)
    idx2 = x2[:, -1].astype(jnp.int32)
    #something is happening

    amplitude = parameters['amplitude']
    lengthscale = parameters['lengthscale']
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
    mask00 = (idx1[:, None] == 0) & (idx2[None, :] == 0)
    mask01 = (idx1[:, None] == 0) & (idx2[None, :] == 1)
    mask02 = (idx1[:, None] == 0) & (idx2[None, :] == 2)
    mask03 = (idx1[:, None] == 0) & (idx2[None, :] == 3)
    mask04 = (idx1[:, None] == 0) & (idx2[None, :] == 4)
    mask05 = (idx1[:, None] == 0) & (idx2[None, :] == 5)
    mask10 = (idx1[:, None] == 1) & (idx2[None, :] == 0)
    mask11 = (idx1[:, None] == 1) & (idx2[None, :] == 1)
    mask12 = (idx1[:, None] == 1) & (idx2[None, :] == 2)
    mask13 = (idx1[:, None] == 1) & (idx2[None, :] == 3)
    mask14 = (idx1[:, None] == 1) & (idx2[None, :] == 4)
    mask15 = (idx1[:, None] == 1) & (idx2[None, :] == 5)
    mask20 = (idx1[:, None] == 2) & (idx2[None, :] == 0)
    mask21 = (idx1[:, None] == 2) & (idx2[None, :] == 1)
    mask22 = (idx1[:, None] == 2) & (idx2[None, :] == 2)
    mask23 = (idx1[:, None] == 2) & (idx2[None, :] == 3)
    mask24 = (idx1[:, None] == 2) & (idx2[None, :] == 4)
    mask25 = (idx1[:, None] == 2) & (idx2[None, :] == 5)
    mask30 = (idx1[:, None] == 3) & (idx2[None, :] == 0)
    mask31 = (idx1[:, None] == 3) & (idx2[None, :] == 1)
    mask32 = (idx1[:, None] == 3) & (idx2[None, :] == 2)
    mask33 = (idx1[:, None] == 3) & (idx2[None, :] == 3)
    mask34 = (idx1[:, None] == 3) & (idx2[None, :] == 4)
    mask35 = (idx1[:, None] == 3) & (idx2[None, :] == 5)
    mask40 = (idx1[:, None] == 4) & (idx2[None, :] == 0)
    mask41 = (idx1[:, None] == 4) & (idx2[None, :] == 1)
    mask42 = (idx1[:, None] == 4) & (idx2[None, :] == 2)
    mask43 = (idx1[:, None] == 4) & (idx2[None, :] == 3)
    mask44 = (idx1[:, None] == 4) & (idx2[None, :] == 4)
    mask45 = (idx1[:, None] == 4) & (idx2[None, :] == 5)
    mask50 = (idx1[:, None] == 5) & (idx2[None, :] == 0)
    mask51 = (idx1[:, None] == 5) & (idx2[None, :] == 1)
    mask52 = (idx1[:, None] == 5) & (idx2[None, :] == 2)
    mask53 = (idx1[:, None] == 5) & (idx2[None, :] == 3)
    mask54 = (idx1[:, None] == 5) & (idx2[None, :] == 4)
    mask55 = (idx1[:, None] == 5) & (idx2[None, :] == 5)
    K = jnp.where(mask00, k00(xx, yy), 0.0)
    K = jnp.where(mask01, k01(xx, yy), K)
    K = jnp.where(mask02, k02(xx, yy), K)
    K = jnp.where(mask03, k03(xx, yy), K)
    K = jnp.where(mask04, k04(xx, yy), K)
    K = jnp.where(mask05, k05(xx, yy), K)
    K = jnp.where(mask10, k10(xx, yy), K)
    K = jnp.where(mask11, k11(xx, yy), K)
    K = jnp.where(mask12, k12(xx, yy), K)
    K = jnp.where(mask13, k13(xx, yy), K)
    K = jnp.where(mask14, k14(xx, yy), K)
    K = jnp.where(mask15, k15(xx, yy), K)
    K = jnp.where(mask20, k20(xx, yy), K)
    K = jnp.where(mask21, k21(xx, yy), K)
    K = jnp.where(mask22, k22(xx, yy), K)
    K = jnp.where(mask23, k23(xx, yy), K)
    K = jnp.where(mask24, k24(xx, yy), K)
    K = jnp.where(mask25, k25(xx, yy), K)
    K = jnp.where(mask30, k30(xx, yy), K)
    K = jnp.where(mask31, k31(xx, yy), K)
    K = jnp.where(mask32, k32(xx, yy), K)
    K = jnp.where(mask33, k33(xx, yy), K)
    K = jnp.where(mask34, k34(xx, yy), K)
    K = jnp.where(mask35, k35(xx, yy), K)
    K = jnp.where(mask40, k40(xx, yy), K)
    K = jnp.where(mask41, k41(xx, yy), K)
    K = jnp.where(mask42, k42(xx, yy), K)
    K = jnp.where(mask43, k43(xx, yy), K)
    K = jnp.where(mask44, k44(xx, yy), K)
    K = jnp.where(mask45, k45(xx, yy), K)
    K = jnp.where(mask50, k50(xx, yy), K)
    K = jnp.where(mask51, k51(xx, yy), K)
    K = jnp.where(mask52, k52(xx, yy), K)
    K = jnp.where(mask53, k53(xx, yy), K)
    K = jnp.where(mask54, k54(xx, yy), K)
    K = jnp.where(mask55, k55(xx, yy), K)
                         
   
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
@jax.jit
def single_mll(params, X, Y, sigma, order_of_parameters):
    # Formula: -0.5 * (y.T @ K_inv @ y + log|K| + N*log(2pi))
                           
    #expects params to be a vector of all parameters of the kernel in the order they are defined in the kernel function.
    p = {}
    for key in order_of_parameters:
        p[key] = params[order_of_parameters[key]]
    
    K = kernel(X, X, p)
    K += sigma**2 * jnp.eye(K.shape[0])
    L = jnp.linalg.cholesky(K)
    L_inv_Y = solve_triangular(L, Y, lower=True) 
    fit_term = -0.5 * jnp.sum(L_inv_Y**2) 
    complexity_term = -jnp.sum(jnp.log(jnp.diagonal(L)))
    return fit_term + complexity_term 

mll = jax.jit(
    jax.vmap(single_mll, in_axes=(0, None, None, None, None))
)                                    