import sympy
from PCGP import PCGP_Builder

def B(D, x):
    nu0 = sympy.symbols("nu0")
    return sympy.matrices.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-nu0*x[2]*D[1]**2-D[2]**2*nu0, x[2]*nu0*D[1]*D[2], nu0*D[0]*D[2]],
        [nu0*x[2]*D[1]*D[0], -nu0*x[2]*D[0]**2-nu0*D[2]**2, nu0*D[1]*D[2]],
        [nu0*D[0]*D[2], nu0*D[2]*D[1], -nu0*D[0]**2-nu0*D[1]**2],
        [0, -D[2], D[1]],
        [D[2], 0, -D[0]],
        [-D[1], D[0], 0]
    ])

builder = PCGP_Builder()
builder.add_kernel(B, number_of_input_dimensions=3, shared_base_kernel = True)
builder.write("Experiment3_PIGP")