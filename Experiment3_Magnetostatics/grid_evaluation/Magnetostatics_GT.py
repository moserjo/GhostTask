
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    nu0 = parameters['nu0']
    amplitude = parameters['amplitude']
    lengthscale = parameters['lengthscale']
    def k00(x, y):
        return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k01(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k02(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k03(x, y):
        return amplitude*nu0*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k04(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k05(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k06(x, y):
        return amplitude*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k07(x, y):
        return amplitude*(x[...,1] - y[...,1])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k10(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k11(x, y):
        return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k12(x, y):
        return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k13(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k14(x, y):
        return amplitude*nu0*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*y[...,2] + (lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k15(x, y):
        return -amplitude*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k16(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k17(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k20(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k21(x, y):
        return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k22(x, y):
        return amplitude*(lengthscale - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k23(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k24(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k25(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k26(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k27(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k30(x, y):
        return amplitude*nu0*((lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] - (x[...,2] - y[...,2])**2) - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k31(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k32(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k33(x, y):
        return amplitude*nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,1] - y[...,1])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*y[...,2] - 2*(lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2] + (2*lengthscale**3 - 2*lengthscale**2*(2*(x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)**2 + 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (3*lengthscale**3 - 3*lengthscale**2*(x[...,1] - y[...,1])**2 + lengthscale*(-(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 + 2*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (6*lengthscale**3 - 18*lengthscale**2*(x[...,1] - y[...,1])**2 + 9*lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
    def k34(x, y):
        return amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (3*lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - (-9*lengthscale**2 + lengthscale*(3*(x[...,0] - y[...,0])**2 + 2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - (-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2))*x[...,2]*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
    def k35(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k36(x, y):
        return -amplitude*nu0*(x[...,2] - y[...,2])*(-3*lengthscale**2 + lengthscale*(2*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k37(x, y):
        return -amplitude*nu0*(x[...,1] - y[...,1])*((lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2) + (14*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k40(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k41(x, y):
        return amplitude*nu0*((lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2) - (-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k42(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k43(x, y):
        return amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (3*lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - (-3*lengthscale**2 + lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + (8*lengthscale**2 - 2*lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2))*x[...,2]*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
    def k44(x, y):
        return amplitude*nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,0] - y[...,0])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2) - (lengthscale - (x[...,1] - y[...,1])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*x[...,2]*y[...,2] + (lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 - (lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2] - (-3*lengthscale**3 + lengthscale**2*(2*(x[...,0] - y[...,0])**2 + 5*(x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 - 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (3*lengthscale**3 - 3*lengthscale**2*(x[...,0] - y[...,0])**2 + lengthscale*(-(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2 + 2*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (3*lengthscale**3 - 3*lengthscale**2*(x[...,0] - y[...,0])**2 + lengthscale*(-(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2 + 2*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (6*lengthscale**3 - 18*lengthscale**2*(x[...,0] - y[...,0])**2 + 9*lengthscale*(lengthscale - (x[...,0] - y[...,0])**2)**2 - (3*lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,0] - y[...,0])**2)*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
    def k45(x, y):
        return amplitude*nu0*(x[...,2] - y[...,2])*(-3*lengthscale**2 + lengthscale*(2*(x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) - (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2 + (-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k46(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k47(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*((lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + 2*(lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2) + (14*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2) + (lengthscale - (x[...,1] - y[...,1])**2)**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k50(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k51(x, y):
        return amplitude*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k52(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k53(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k54(x, y):
        return amplitude*nu0*(x[...,2] - y[...,2])*(3*lengthscale**2 - lengthscale*(3*(x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,0] - y[...,0])**2 + (3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])**2)*y[...,2] + (lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k55(x, y):
        return amplitude*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k56(x, y):
        return -amplitude*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k57(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k60(x, y):
        return -amplitude*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k61(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k62(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k63(x, y):
        return -amplitude*nu0*(x[...,2] - y[...,2])*(3*lengthscale**2 - lengthscale*(3*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k64(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k65(x, y):
        return -amplitude*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k66(x, y):
        return amplitude*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k67(x, y):
        return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k70(x, y):
        return -amplitude*(x[...,1] - y[...,1])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k71(x, y):
        return amplitude*(x[...,0] - y[...,0])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k72(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k73(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(3*lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (-lengthscale + (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale + 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2) + (2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k74(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(3*lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,2] - y[...,2])**2) - (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] - (x[...,2] - y[...,2])**2) - (-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2)*y[...,2] + (2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2)*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2))*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
    def k75(x, y):
        return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k76(x, y):
        return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k77(x, y):
        return amplitude*(4*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,1] - y[...,1])**2)**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    function_grid = [
        [k00, k01, k02, k03, k04, k05, k06, k07],
        [k10, k11, k12, k13, k14, k15, k16, k17],
        [k20, k21, k22, k23, k24, k25, k26, k27],
        [k30, k31, k32, k33, k34, k35, k36, k37],
        [k40, k41, k42, k43, k44, k45, k46, k47],
        [k50, k51, k52, k53, k54, k55, k56, k57],
        [k60, k61, k62, k63, k64, k65, k66, k67],
        [k70, k71, k72, k73, k74, k75, k76, k77],
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
                