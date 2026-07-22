from PCGP import PCGP_Builder_jax
import sympy as sp


def B(D, x):
    a = sp.Symbol("a")
    return sp.matrices.Matrix([[-a*D[1]*D[0] + x[0]*D[0]+ D[1] -1],
                               [x[0]*D[1]*D[0]+D[1]**2-D[1]],
                               [x[0]**2*D[0]**2+2*x[0]*D[1]*D[0]+D[1]**2-D[1]]
                                ])


builder1 = PCGP_Builder_jax()
builder1.add_kernel(B,  number_of_input_dimensions=2, shared_base_kernel = True)
builder1.write("Pedagogical_GT")

def B(D, x):
    a = sp.Symbol("a")
    return sp.matrices.Matrix([
                               [1, 0],
                               [0, 1],
                               [x[0]*D[0]+D[1], D[0]*a],
                               [0, x[0]*D[0]+D[1]],
                               ])



builder2 = PCGP_Builder_jax()
builder2.add_kernel(B,  number_of_input_dimensions=2, shared_base_kernel = True)
builder2.write("Pedagogical_PIGP")