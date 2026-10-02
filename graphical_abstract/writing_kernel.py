from PCGP import PCGP_Builder
import sympy


def B(X, t):
    g, length = sympy.symbols("g, length")
    return sympy.matrices.Matrix([[1         ]   ,    
                                  [1          ]   ,  
                                  [1           ]   , 
                                  [X[0]**2*length+g],
                                 
                                  ])



builder2 = PCGP_Builder()
builder2.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder2.write("Tripendulum_reference")