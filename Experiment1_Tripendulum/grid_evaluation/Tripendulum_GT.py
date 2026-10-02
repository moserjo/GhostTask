
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    amplitude = parameters['amplitude']
    length = parameters['length']
    lengthscale = parameters['lengthscale']
    g = parameters['g']
    def k00(x, y):
        return amplitude*(-2*g*length**3*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**4*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2) + lengthscale**4*(g**2*length**2 + 1))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k01(x, y):
        return amplitude*(g**2*length**2*lengthscale**4*(x[...,0] - y[...,0]) - 2*g*length**3*lengthscale**2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + length**4*(x[...,0] - y[...,0])*(12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)) + lengthscale**5)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**5
    def k02(x, y):
        return amplitude*(-2*g*length**3*lengthscale**3*(x[...,0] - y[...,0]) - 2*g*length**3*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2)*y[...,0] + 2*length**4*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + length**4*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*y[...,0] + lengthscale**4*(g**2*length**2*y[...,0] + 1))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k03(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k04(x, y):
        return -amplitude*length*(g**3*lengthscale**6 - 3*g**2*length*lengthscale**4*(lengthscale - (x[...,0] - y[...,0])**2) + g*length**2*lengthscale**2*(7*lengthscale**2 - 14*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)**2 + (x[...,0] - y[...,0])**4) - length**3*(12*lengthscale**3 - 12*lengthscale**2*(x[...,0] - y[...,0])**2 - 8*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**6
    def k10(x, y):
        return -amplitude*(g**2*length**2*lengthscale**4*(x[...,0] - y[...,0]) - 2*g*length**3*lengthscale**2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + length**4*(x[...,0] - y[...,0])*(12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)) - lengthscale**5)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**5
    def k11(x, y):
        return amplitude*(g**2*length**2*lengthscale**4*(lengthscale - (x[...,0] - y[...,0])**2) + 2*g*length**3*lengthscale**2*(-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2) + length**4*(6*lengthscale**3 - 18*lengthscale**2*(x[...,0] - y[...,0])**2 + 9*lengthscale*(lengthscale - (x[...,0] - y[...,0])**2)**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,0] - y[...,0])**2) + lengthscale**6)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**6
    def k12(x, y):
        return -amplitude*(g**2*length**2*lengthscale**4*(x[...,0] - y[...,0])*y[...,0] + 2*g*length**3*lengthscale**3*(lengthscale - (x[...,0] - y[...,0])**2) - 2*g*length**3*lengthscale**2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*y[...,0] + 2*length**4*lengthscale*(-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2) + length**4*(x[...,0] - y[...,0])*(12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2))*y[...,0] - lengthscale**5)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**5
    def k13(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k14(x, y):
        return amplitude*length*(x[...,0] - y[...,0])*(g**3*lengthscale**6 - 3*g**2*length*lengthscale**4*(3*lengthscale - (x[...,0] - y[...,0])**2) + g*length**2*lengthscale**2*(39*lengthscale**2 - 22*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2) + (x[...,0] - y[...,0])**4) + length**3*(-60*lengthscale**3 + 36*lengthscale**2*(x[...,0] - y[...,0])**2 - 4*lengthscale*(2*(-3*lengthscale + (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)**2) - (3*lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**7
    def k20(x, y):
        return amplitude*(2*g*length**3*lengthscale**3*(x[...,0] - y[...,0]) - 2*g*length**3*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2)*x[...,0] - 2*length**4*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + length**4*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*x[...,0] + lengthscale**4*(g**2*length**2*x[...,0] + 1))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k21(x, y):
        return amplitude*(g**2*length**2*lengthscale**4*(x[...,0] - y[...,0])*x[...,0] - 2*g*length**3*lengthscale**3*(lengthscale - (x[...,0] - y[...,0])**2) - 2*g*length**3*lengthscale**2*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*x[...,0] - 2*length**4*lengthscale*(-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2) + length**4*(x[...,0] - y[...,0])*(12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2))*x[...,0] + lengthscale**5)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**5
    def k22(x, y):
        return amplitude*(-2*g*length**3*lengthscale**3*(x[...,0] - y[...,0])**2 + 2*length**4*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + length**4*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*x[...,0]*y[...,0] - 2*length**3*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2)*(g*x[...,0]*y[...,0] - 2*length) + lengthscale**4*(g**2*length**2*x[...,0]*y[...,0] + 1))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k23(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k24(x, y):
        return -amplitude*length*(g**3*lengthscale**6*x[...,0] + 2*g**2*length*lengthscale**5*(x[...,0] - y[...,0]) - 3*g**2*length*lengthscale**4*(lengthscale - (x[...,0] - y[...,0])**2)*x[...,0] - 4*g*length**2*lengthscale**3*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + g*length**2*lengthscale**2*(7*lengthscale**2 - 14*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)**2 + (x[...,0] - y[...,0])**4)*x[...,0] + 2*length**3*lengthscale*(x[...,0] - y[...,0])*(15*lengthscale**2 - 10*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4) - length**3*(12*lengthscale**3 - 12*lengthscale**2*(x[...,0] - y[...,0])**2 - 8*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4))*x[...,0])*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**6
    def k30(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k31(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k32(x, y):
        return amplitude*(g*lengthscale**2 - length*(lengthscale - (x[...,0] - y[...,0])**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    def k33(x, y):
        return amplitude*(g**2*lengthscale**4 - 2*g*length*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2) + length**2*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
    def k34(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k40(x, y):
        return -amplitude*length*(g**3*lengthscale**6 - 3*g**2*length*lengthscale**4*(lengthscale - (x[...,0] - y[...,0])**2) + g*length**2*lengthscale**2*(7*lengthscale**2 - 14*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)**2 + (x[...,0] - y[...,0])**4) - length**3*(12*lengthscale**3 - 12*lengthscale**2*(x[...,0] - y[...,0])**2 - 8*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**6
    def k41(x, y):
        return -amplitude*length*(x[...,0] - y[...,0])*(g**3*lengthscale**6 - 3*g**2*length*lengthscale**4*(3*lengthscale - (x[...,0] - y[...,0])**2) + g*length**2*lengthscale**2*(39*lengthscale**2 - 22*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2) + (x[...,0] - y[...,0])**4) - length**3*(60*lengthscale**3 - 36*lengthscale**2*(x[...,0] - y[...,0])**2 + 12*lengthscale*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2) + (3*lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**7
    def k42(x, y):
        return amplitude*length*(-g**3*lengthscale**6*y[...,0] + 2*g**2*length*lengthscale**5*(x[...,0] - y[...,0]) + 3*g**2*length*lengthscale**4*(lengthscale - (x[...,0] - y[...,0])**2)*y[...,0] - 4*g*length**2*lengthscale**3*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) - g*length**2*lengthscale**2*(7*lengthscale**2 - 14*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)**2 + (x[...,0] - y[...,0])**4)*y[...,0] + 2*length**3*lengthscale*(x[...,0] - y[...,0])*(15*lengthscale**2 - 10*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4) + length**3*(12*lengthscale**3 - 12*lengthscale**2*(x[...,0] - y[...,0])**2 - 8*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4))*y[...,0])*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**6
    def k43(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k44(x, y):
        return amplitude*(g**4*lengthscale**8 - 4*g**3*length*lengthscale**6*(lengthscale - (x[...,0] - y[...,0])**2) + 2*g**2*length**2*lengthscale**4*(7*lengthscale**2 - 14*lengthscale*(x[...,0] - y[...,0])**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)**2 + (x[...,0] - y[...,0])**4) - 4*g*length**3*lengthscale**2*(12*lengthscale**3 - 12*lengthscale**2*(x[...,0] - y[...,0])**2 - 8*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)) + length**4*(24*lengthscale**4 - 96*lengthscale**3*(x[...,0] - y[...,0])**2 + 72*lengthscale**2*(lengthscale - (x[...,0] - y[...,0])**2)**2 - 16*lengthscale*(3*lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,0] - y[...,0])**2 + (3*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (x[...,0] - y[...,0])**4)**2))*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**8
    function_grid = [
        [k00, k01, k02, k03, k04],
        [k10, k11, k12, k13, k14],
        [k20, k21, k22, k23, k24],
        [k30, k31, k32, k33, k34],
        [k40, k41, k42, k43, k44],
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
                