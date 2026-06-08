from PCGP import PCGP_Builder
import sympy as sp


def B(D, x):
    a = sp.Symbol("a")
    return sp.matrices.Matrix([
                               [1, 0],
                               [0, 1],
                               [x[0]*D[0]+D[1], D[0]*a],
                               [0, x[0]*D[0]+D[1]],
                               ])
builder = PCGP_Builder()
builder.add_kernel(B, number_of_input_dimensions=2, shared_base_kernel = True)
builder.write("Experiment1_Pedagogical_PIGP")
