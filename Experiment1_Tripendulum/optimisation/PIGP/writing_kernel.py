from PCGP import PCGP_Builder
import sympy


def B(D, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1, 0, 0],
                                  [0,1, 0],
                                  [0, 0, 1],
                                  [length*D[0]**2 + g, 0, 0],
                                  [0, length*D[0]**2 + g, 0],
                                  [0, 0, length*D[0]**2 + g]
                                  ])


builder = PCGP_Builder()
builder.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder.write("Tripendulum_PIGP")

def B(D, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1, 0, 0],
                                  [0,1, 0],
                                  [0, 0, 1],
                                  [length*D[0]**2 + g, 0, 0],
                                  [0, length*D[0]**2 + g, 0],
                                  [0, 0, length*D[0]**2 + g],
                                  [D[0], 0, 0],
                                  [0,D[0], 0],
                                  [0, 0, D[0]],
                                  ])


builder = PCGP_Builder()
builder.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder.write("Tripendulum_PIGP_forward")