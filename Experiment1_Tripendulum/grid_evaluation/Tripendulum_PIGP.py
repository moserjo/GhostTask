
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    g = parameters['g']
    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    length = parameters['length']
    def k00(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
    def k01(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k02(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k03(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k04(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k05(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k10(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k11(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
    def k12(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k13(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k14(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k15(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k20(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k21(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k22(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
    def k23(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k24(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k25(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k30(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k31(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k32(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k33(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k34(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k35(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k40(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k41(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k42(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k43(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k44(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k45(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k50(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k51(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k52(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k53(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k54(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k55(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    function_grid = [
        [k00, k01, k02, k03, k04, k05],
        [k10, k11, k12, k13, k14, k15],
        [k20, k21, k22, k23, k24, k25],
        [k30, k31, k32, k33, k34, k35],
        [k40, k41, k42, k43, k44, k45],
        [k50, k51, k52, k53, k54, k55],
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
                