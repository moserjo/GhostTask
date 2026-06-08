from PCGP import PCGP_Builder
import sympy


def B(X, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1/length, 0, 0],
                                  [0,1/length, 0],
                                  [0, 0, 1/length],
                                  [X[0]**2 + g/length, 0, 0],
                                  [0, X[0]**2 + g/length, 0],
                                  [0, 0, X[0]**2 + g/length]
                                  ])


builder = PCGP_Builder()
builder.add_kernel(B,  number_of_input_dimensions=1)
builder.write("Tripendulum_PIGP_no_shared_kernel")

builder2 = PCGP_Builder()
builder2.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder2.write("Tripendulum_PIGP_shared_kernel")