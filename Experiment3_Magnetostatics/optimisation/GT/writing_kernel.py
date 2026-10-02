import sympy
from PCGP import PCGP_Builder

def B(D, x):
    nu0 = sympy.symbols("nu0")
    return sympy.matrices.Matrix([
       [-D[0], -D[1]],
       [-D[1],  D[0]],
       [-D[2],  0],
       [0,      nu0*D[1]*(D[2]**2+x[2]*(D[0]**2+D[1]**2))],
       [0,      -nu0*D[0]*(D[2]**2+x[2]*(D[0]**2+D[1]**2))],
       [0,      -D[2]*D[0]],
       [0,      -D[2]*D[1]],
       [0,      (D[0]**2+D[1]**2)]

       ])


builder = PCGP_Builder()
builder.add_kernel(B, number_of_input_dimensions=3, shared_base_kernel = True)
builder.write("Experiment3_GT")
