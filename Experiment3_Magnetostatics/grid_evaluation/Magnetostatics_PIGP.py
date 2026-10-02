
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    amplitude = parameters['amplitude']
    nu0 = parameters['nu0']
    lengthscale = parameters['lengthscale']
    def k00(x, y):
        return amplitude*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)
    def k01(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k02(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k03(x, y):
        return amplitude*nu0*(lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k04(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**2
    def k05(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k06(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k07(x, y):
        return amplitude*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k08(x, y):
        return -amplitude*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k10(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k11(x, y):
        return amplitude*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)
    def k12(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k13(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**2
    def k14(x, y):
        return amplitude*nu0*(lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k15(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k16(x, y):
        return -amplitude*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k17(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k18(x, y):
        return amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k20(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k21(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k22(x, y):
        return amplitude*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)
    def k23(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k24(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k25(x, y):
        return -amplitude*nu0*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k26(x, y):
        return amplitude*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k27(x, y):
        return -amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k28(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k30(x, y):
        return amplitude*nu0*(lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k31(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**2
    def k32(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k33(x, y):
        return amplitude*nu0**2*(3*lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] + y[...,2]) + (lengthscale - (x[...,2] - y[...,2])**2)**2 + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 + (lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2]*y[...,2] + (2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2)*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k34(x, y):
        return amplitude*nu0**2*(x[...,1] - y[...,1])*((lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*x[...,2]*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*y[...,2] - (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,2]*y[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k35(x, y):
        return -amplitude*nu0**2*((x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-7*lengthscale - (lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - (lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k36(x, y):
        return amplitude*nu0*((lengthscale - (x[...,2] - y[...,2])**2)*x[...,2] + (x[...,0] - y[...,0])*(x[...,2] - y[...,2]))*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k37(x, y):
        return amplitude*nu0*(x[...,2] - y[...,2])*(4*lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,0] - y[...,0])**2 - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k38(x, y):
        return -amplitude*nu0*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,0] - y[...,0])*(x[...,2] - y[...,2])*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k40(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**2
    def k41(x, y):
        return amplitude*nu0*(lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k42(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k43(x, y):
        return amplitude*nu0**2*(x[...,1] - y[...,1])*((lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*x[...,2]*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*x[...,2] - (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*x[...,2]*y[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k44(x, y):
        return amplitude*nu0**2*(3*lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] + y[...,2]) + (lengthscale - (x[...,2] - y[...,2])**2)**2 + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 + (lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*x[...,2]*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k45(x, y):
        return -amplitude*nu0**2*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-7*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k46(x, y):
        return -amplitude*nu0*(x[...,2] - y[...,2])*(4*lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] - (x[...,1] - y[...,1])**2 - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k47(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - 1)*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k48(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k50(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k51(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k52(x, y):
        return -amplitude*nu0*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k53(x, y):
        return -amplitude*nu0**2*((x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-7*lengthscale - (lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - (lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k54(x, y):
        return -amplitude*nu0**2*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-7*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k55(x, y):
        return amplitude*nu0**2*(6*lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - 4*lengthscale*(x[...,0] - y[...,0])**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,1] - y[...,1])**2)**2 + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
    def k56(x, y):
        return -amplitude*nu0*(x[...,1] - y[...,1])*(-5*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k57(x, y):
        return amplitude*nu0*(x[...,0] - y[...,0])*(-5*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k58(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k60(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k61(x, y):
        return amplitude*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k62(x, y):
        return -amplitude*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k63(x, y):
        return -amplitude*nu0*((lengthscale - (x[...,2] - y[...,2])**2)*y[...,2] + (x[...,0] - y[...,0])*(x[...,2] - y[...,2]))*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k64(x, y):
        return amplitude*nu0*(x[...,2] - y[...,2])*(4*lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] - (x[...,1] - y[...,1])**2 - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k65(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(-5*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k66(x, y):
        return -amplitude*(-2*lengthscale + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k67(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k68(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k70(x, y):
        return -amplitude*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k71(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k72(x, y):
        return amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k73(x, y):
        return -amplitude*nu0*(x[...,2] - y[...,2])*(4*lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,0] - y[...,0])**2 - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k74(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(y[...,2] - 1)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k75(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(-5*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k76(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k77(x, y):
        return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k78(x, y):
        return amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k80(x, y):
        return amplitude*(x[...,1] - y[...,1])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k81(x, y):
        return -amplitude*(x[...,0] - y[...,0])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale
    def k82(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k83(x, y):
        return amplitude*nu0*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,0] - y[...,0])*(x[...,2] - y[...,2])*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k84(x, y):
        return -amplitude*nu0*(x[...,0] - y[...,0])*(lengthscale + (lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
    def k85(x, y):
        return jnp.zeros_like(x[...,0]-y[...,0])
    def k86(x, y):
        return amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k87(x, y):
        return amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    def k88(x, y):
        return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*jnp.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
    function_grid = [
        [k00, k01, k02, k03, k04, k05, k06, k07, k08],
        [k10, k11, k12, k13, k14, k15, k16, k17, k18],
        [k20, k21, k22, k23, k24, k25, k26, k27, k28],
        [k30, k31, k32, k33, k34, k35, k36, k37, k38],
        [k40, k41, k42, k43, k44, k45, k46, k47, k48],
        [k50, k51, k52, k53, k54, k55, k56, k57, k58],
        [k60, k61, k62, k63, k64, k65, k66, k67, k68],
        [k70, k71, k72, k73, k74, k75, k76, k77, k78],
        [k80, k81, k82, k83, k84, k85, k86, k87, k88],
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
                