from PCGP import PCGP_Builder_jax
import sympy


def B(X, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1,                  -length**2*X[0]**2-g*length],
                                  [1,                  -X[0]**3*length**2-X[0]*g*length],
                                  [1,                  -t[0]*length**2*X[0]**2+2*length**2*X[0]-g*length*t[0]],
                                  [X[0]**2*length+g,   0],
                                  [0,                  X[0]**4*length**2+2*X[0]**2*g*length+g**2]
                                  ])




builder1 = PCGP_Builder_jax()
builder1.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder1.write("Tripendulum_GT")

def B(X, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1/length, 0, 0],
                                  [0,1/length, 0],
                                  [0, 0, 1/length],
                                  [X[0]**2 + g/length, 0, 0],
                                  [0, X[0]**2 + g/length, 0],
                                  [0, 0, X[0]**2 + g/length]
                                  ])



builder2 = PCGP_Builder_jax()
builder2.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder2.write("Tripendulum_PIGP")