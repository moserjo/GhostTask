
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    a = parameters['a']
    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    def k00(x, y):
        return amplitude*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)
    def k01(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k02(x, y):
        return amplitude*((x[...,0] - y[...,0])*y[...,0] + x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k03(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k10(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k11(x, y):
        return amplitude*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)
    def k12(x, y):
        return a*amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k13(x, y):
        return amplitude*((x[...,0] - y[...,0])*y[...,0] + x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k20(x, y):
        return -amplitude*((x[...,0] - y[...,0])*x[...,0] + x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k21(x, y):
        return -a*amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k22(x, y):
        return amplitude*(a**2*(lengthscale - (x[...,0] - y[...,0])**2) + lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,0]*y[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] - (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**2
    def k23(x, y):
        return a*amplitude*((lengthscale - (x[...,0] - y[...,0])**2)*y[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1]))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**2
    def k30(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k31(x, y):
        return -amplitude*((x[...,0] - y[...,0])*x[...,0] + x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale
    def k32(x, y):
        return a*amplitude*((lengthscale - (x[...,0] - y[...,0])**2)*x[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1]))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**2
    def k33(x, y):
        return -amplitude*(-lengthscale - (lengthscale - (x[...,0] - y[...,0])**2)*x[...,0]*y[...,0] + (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] + (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**2
    function_grid = [
        [k00, k01, k02, k03],
        [k10, k11, k12, k13],
        [k20, k21, k22, k23],
        [k30, k31, k32, k33],
    ]
    for i, idxs1 in enumerate(structure.x1):
        for j, idxs2 in enumerate(structure.x2):
            x_block = xx[idxs1]
            y_block = yy[idxs2]
            block = function_grid[i][j](x_block[:, None, :], y_block[None, :, :])
            K = K.at[jnp.ix_(idxs1, idxs2)].set(block)

    return K


def kernel(x1, x2, parameters, structure):
    k = (
        kernel_0(x1, x2, parameters, structure)
        ) 
    return k
                