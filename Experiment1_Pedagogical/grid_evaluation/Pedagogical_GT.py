
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    a = parameters['a']
    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    def k00(x, y):
        return amplitude*(a**2*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - a*lengthscale*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + lengthscale**4 + lengthscale**3*(x[...,0] - y[...,0])**2 + lengthscale**2*(2*a*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,0]*y[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] - (x[...,1] - y[...,1])**2))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k01(x, y):
        return amplitude*(a*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) - a*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,0] + lengthscale**3*(x[...,1] - y[...,1]) + lengthscale**2*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1]) + lengthscale*(-a*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]*y[...,0] + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k02(x, y):
        return amplitude*(a*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,0]**2 - (x[...,1] - y[...,1])**2) - 2*a*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,0] + lengthscale**3*(x[...,1] - y[...,1]) + lengthscale**2*((lengthscale - (x[...,0] - y[...,0])**2)*y[...,0]**2 + (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - 2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0]) + lengthscale*(-a*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0]) + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]*y[...,0] + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*y[...,0]**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0] + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*x[...,0]*y[...,0]**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k10(x, y):
        return -amplitude*(-a*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + a*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0] + lengthscale**3*(x[...,1] - y[...,1]) + lengthscale**2*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1]) + lengthscale*(-a*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]*y[...,0] + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k11(x, y):
        return amplitude*(lengthscale**2*(lengthscale - (x[...,1] - y[...,1])**2) + 2*lengthscale**2 + lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,0] + y[...,0])*(x[...,1] - y[...,1]) + (lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0]*y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k12(x, y):
        return amplitude*(lengthscale**2*(lengthscale - (x[...,1] - y[...,1])**2) + 2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 - lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*y[...,0]**2 - (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,0]**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0]*y[...,0]**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] + 2*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0]*y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k20(x, y):
        return -amplitude*(-a*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,0]**2 - (x[...,1] - y[...,1])**2) + 2*a*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0] + lengthscale**3*(x[...,1] - y[...,1]) - lengthscale**2*((lengthscale - (x[...,0] - y[...,0])**2)*x[...,0]**2 - 2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] + (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0]) + lengthscale*(-a*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]*y[...,0] + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0] + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*x[...,0]**2*y[...,0] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])))*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k21(x, y):
        return amplitude*(lengthscale**2*(lengthscale - (x[...,1] - y[...,1])**2) + 2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]**2 + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] - (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,0]**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0]**2*y[...,0] - 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] + 2*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0]*y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    def k22(x, y):
        return amplitude*(lengthscale**2*(lengthscale - (x[...,1] - y[...,1])**2) + 2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*x[...,0]**2 - (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*y[...,0]**2 + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,0] - 2*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*y[...,0]) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,0]**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,0]**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2 - 2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0]**2*y[...,0] - 2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0]*y[...,0]**2 - 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,0] - 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,0] + 4*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,0]*y[...,0] + (2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*x[...,0]**2*y[...,0]**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)/lengthscale)/lengthscale**4
    function_grid = [
        [k00, k01, k02],
        [k10, k11, k12],
        [k20, k21, k22],
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
                