import sympy
from PCGP import PCGP_Builder

def B(D, x):
    nu = sympy.symbols("nu0")
    return sympy.matrices.Matrix([
       [-D[0],  0,   0]     ,
       [-D[1],  -x[2]*D[2] - 2,   D[2]**2]     ,
       [-D[2],  x[2]**2*D[1],            -x[2]*D[1]*D[2] + D[1]],
       
       [0,      -(x[2]**2*(D[0]**2+D[1]**2)+x[2]*D[2]**2+3*D[2])*D[1]*nu, ((x[2]*D[2]-1)*D[1]**2+x[2]*D[2]*D[0]**2+D[2]**3-D[0]**2)*D[1]*nu],
       [0,      (x[2]**2*D[2]*(D[0]**2+D[1]**2)+x[2]*(D[2]**3+2*D[0]**2+2*D[1]**2) + 4*D[2]**2)*nu, -((D[0]**2+D[1]**2)*x[2]+D[2]**2)*D[2]**2*nu],
       
       [0,  nu*(x[2]**2*D[1]**2+x[2]*D[2]**2+3*D[2]),   -nu*((x[2]*D[2]-1)*D[1]**2+D[2]**3)],
       [0,  -nu*D[0]*D[1]*x[2]**2, nu*D[0]*D[1]*(x[2]*D[2]-1)],
       [0,  -x[2]*D[0]*nu*(x[2]*D[2]+2),    x[2]*D[0]*D[2]**2*nu],
       ])


#builder = PCGP_Builder()
#builder.add_kernel(B, number_of_input_dimensions=3, shared_base_kernel = True)
#builder.write("Experiment3_analytic")

def B2(D, x):
    nu = sympy.symbols("nu0")
    return sympy.matrices.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-nu*x[2]*D[1]**2-D[2]**2*nu, x[2]*nu*D[1]*D[2], nu*D[0]*D[2]],
        [nu*x[2]*D[1]*D[0], -nu*x[2]*D[0]**2-nu*D[2]**2, nu*D[1]*D[2]],
        [nu*D[0]*D[2], nu*D[2]*D[1], -nu*D[0]**2-nu*D[1]**2],
        [0, -D[2], D[1]],
        [D[2], 0, -D[0]],
        [-D[1], D[0], 0]
    ])

builder2 = PCGP_Builder()
builder2.add_kernel(B2, number_of_input_dimensions=3, shared_base_kernel = True)
builder2.write("Experiment3_analytic_PIGP")
