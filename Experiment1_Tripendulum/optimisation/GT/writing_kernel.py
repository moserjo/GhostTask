from PCGP import PCGP_Builder
import sympy


def B(D, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1,                  -length**2*D[0]**2-g*length],
                                  [1,                  -D[0]**3*length**2-D[0]*g*length],
                                  [1,                  -t[0]*length**2*D[0]**2+2*length**2*D[0]-g*length*t[0]],
                                  [D[0]**2*length+g,   0],
                                  [0,                  D[0]**4*length**2+2*D[0]**2*g*length+g**2]
                                  ])



builder = PCGP_Builder()
builder.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder.write("Tripendulum_GT")

def B(D, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1,                  -length**2*D[0]**2-g*length],
                                  [1,                  -D[0]**3*length**2-D[0]*g*length],
                                  [1,                  -t[0]*length**2*D[0]**2+2*length**2*D[0]-g*length*t[0]],
                                  [D[0]**2*length+g,   0],
                                  [0,                  D[0]**4*length**2+2*D[0]**2*g*length+g**2],
                                  [D[0],                  D[0]*(-length**2*D[0]**2-g*length)],
                                  [D[0],                  D[0]*(-D[0]**3*length**2-D[0]*g*length)],
                                  [D[0],                  -t[0]*length**2*D[0]**3-length**2*D[0]**2+2*length**2*D[0]**2-g*length*t[0]*D[0]-g*length],
                                  ])



builder = PCGP_Builder()
builder.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder.write("Tripendulum_GT_forward")

