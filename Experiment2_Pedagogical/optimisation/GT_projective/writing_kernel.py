from PCGP import PCGP_Builder
import sympy as sp


def B(D, xx):
    a = sp.Symbol("a")
    (x, t) = xx
    (dx, dt) = D
    return sp.matrices.Matrix([[a*x*dx**2*t**2+a*t**2*dt*dx-2*a*x*dx**2*t-x**2*dx**2*t-2*a*t*dt*dx+a*dx*t**2-2*x*dx*t*dt+2*x**2*dx**2-2*a*dx*t-t*dt**2+4*x*dx*dt+4*a*dx+2*dt**2+t*dt-3*x*dx-5*dt+1],
    [-t**2*x**2*dx**2-2*t**2*x*dt*dx+2*t*x**2*dx**2-t**2*dt**2+4*t*x*dx*dt+2*t*dt**2+t**2*dt-2*x*t*dx-4*t*dt-2*x*dx-2*dt+4],
    [-x**3*dx**3*t-3*t*x**2*dt*dx**2+2*x**3*dx**3-3*t*x*dt**2*dx+6*x**2*dt*dx**2-2*t*x**2*dx**2-t*dt**3+6*x*dx*dt**2-t*x*dx*dt+2*dt**3+t*dt**2-6*x*dx*dt-6*dt**2-2*x*dx+2*dt]])
   
builder = PCGP_Builder()
builder.add_kernel(B, number_of_input_dimensions=2, shared_base_kernel = True)
builder.write("Experiment2_Pedagogical_GT_proj")
