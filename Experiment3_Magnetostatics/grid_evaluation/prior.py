import jax.numpy as jnp
import jax

def single_prior(params, x,  kernel, sigma):
    """ calculates log|K| """
    K = kernel(x, x, params)
    N = K.shape[0]
    K += sigma**2*jnp.eye(N)
    #print(sigma**2*jnp.eye(N))
    L = jnp.linalg.cholesky(K)
    """    eigenvalues, eigenvectors = jnp.linalg.eigh(K)
        eigenvalues = jnp.maximum(eigenvalues, 1e-10)
        Qy = eigenvectors.T @ train_y
        fit_term = -0.5 * jnp.sum(Qy**2 / eigenvalues)
        complexity_term = -0.5 * jnp.sum(jnp.log(eigenvalues))"""
  
  
    complexity_term = -jnp.sum(jnp.log(jnp.diagonal(L)))

    return complexity_term

log_prior = jax.vmap(single_prior, in_axes=(0, None, None, None))


