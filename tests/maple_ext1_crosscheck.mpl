restart:
read "../Maple/ghosttask_projectivity.mpl":

with(Ore_algebra):
with(LinearAlgebra):

assert_true := proc(label, condition)
    if not condition then
        error cat("FAIL: ", label)
    end if:
    printf("PASS: %s\n", label)
end proc:

# Independent symbolic cross-check of the Macaulay2 extension-module
# certificates in the operator-order convention.  The witnesses below
# were computed by the corrected Macaulay2 layer
# (Macaulay2/ext1_constructive.m2); Maple re-verifies the right-inverse
# identities P*S = I through Ore_algebra:-skew_product with the physical
# parameters kept symbolic.  Maple's skew product composes operators in
# the written order, so a passing check certifies the paper's actual
# stably-free condition.  Row injectivity is certified on the Macaulay2
# side by syzygies of the plain transpose; the bounded row-relation
# search here is the independent bounded counterpart.

# ------------------------------------------------------------------
# Experiment 1: shear transport, D = Weyl(t, x) with parameter a.
# Lambda_1 = (-1, -t)^T, one ghost column, exact minimum.
# ------------------------------------------------------------------
A1 := diff_algebra([dt, t], [dx, x], polynom = {t, x}, comm = {a}):

P1 := Matrix([[x*dx+dt, dx*a, 1],
  [0, x*dx+dt, t]]):
S1 := Matrix([[4*t*x*dx^2*a-4*x^2*dx^2-t^2*dx*a+4*t*dt*dx*a+t*x*dx-8*x*dt*dx+2*t*dx*a+t*dt-4*dt^2+2*x*dx-8*dx*a-t+6*dt+1, -2*t*x*dx^2*a+2*x^2*dx^2-2*t*dt*dx*a+4*x*dt*dx-t*dx*a+2*dt^2-x*dx+4*dx*a-3*dt+1],
  [-4*t*x^2*dx^2+t^2*x*dx-8*t*x*dt*dx+t^2*dt-4*t*dt^2+2*t*x*dx-t^2+6*t*dt+4*x*dx+4*dt-6, 2*t*x^2*dx^2+4*t*x*dt*dx+2*t*dt^2-t*x*dx-3*t*dt-2*x*dx+t-2*dt+3],
  [4*x^3*dx^3-t*x^2*dx^2+12*x^2*dt*dx^2-2*t*x*dt*dx+12*x*dt^2*dx+6*x^2*dx^2-t*dt^2+4*dt^3+t*dt-6*dt^2-4*x*dx-2*dt+2, -2*x^3*dx^3-6*x^2*dt*dx^2-6*x*dt^2*dx-3*x^2*dx^2-2*dt^3+3*dt^2-dt]]):

row_relation_1 := GT_row_relation_search(P1, [0, 1, -1, t, -t, dt, -dt], A1):
oracle1 := GT_projectivity_proof_oracle(P1, S1, A1,
    evalb(row_relation_1["relationFound"] = false)):
assert_true("Ex1 operator-order identity P1*S1 = I", oracle1["rightInverseIdentity"]):
assert_true("Ex1 no bounded row relation", oracle1["rowInjective"]):
assert_true("Ex1 projectivity certificate", oracle1["projective"]):

# ------------------------------------------------------------------
# Experiment 2: tripendulum, D = Weyl(t) with symbolic parameters l, g
# treated as invertible constants (coefficient field Q(l, g)).
# Lambda_2 = (-1, -t, 0)^T, one ghost column, exact minimum over Q(l,g).
# ------------------------------------------------------------------
A2 := diff_algebra([dt, t], polynom = {t}, comm = {l, g}):

P2 := Matrix([[l*dt^2+g, 0, 0, -1, 1],
  [0, l*dt^2+g, 0, -1, t],
  [0, 0, l*dt^2+g, -1, 0]]):
S2 := Matrix([[(1/(2*g))*t*dt+1/(2*g), -(1/(2*g))*dt, -(1/(2*g))*t*dt+(1/(2*g))*dt-1/(2*g)],
  [(1/(2*g))*t^2*dt-(1/(2*g))*t, -(1/(2*g))*t*dt+1/g, -(1/(2*g))*t^2*dt+(1/(2*g))*t*dt+(1/(2*g))*t-1/g],
  [0, 0, 0],
  [0, 0, -1],
  [-(l/(2*g))*t*dt^3-(1/2)*t*dt-(3*l/(2*g))*dt^2+1/2, (l/(2*g))*dt^3+(1/2)*dt, (l/(2*g))*t*dt^3-(l/(2*g))*dt^3+(1/2)*t*dt+(3*l/(2*g))*dt^2-(1/2)*dt-1/2]]):

row_relation_2 := GT_row_relation_search(P2, [0, 1, -1, t, -t, dt, -dt], A2):
oracle2 := GT_projectivity_proof_oracle(P2, S2, A2,
    evalb(row_relation_2["relationFound"] = false)):
assert_true("Ex2 operator-order identity P2*S2 = I", oracle2["rightInverseIdentity"]):
assert_true("Ex2 no bounded row relation", oracle2["rowInjective"]):
assert_true("Ex2 projectivity certificate", oracle2["projective"]):

# ------------------------------------------------------------------
# Experiment 3: anisotropic magnetostatics, D = Weyl(x, y, z), nu0 = 1.
# Two-column block; the constant witness uses the J_z source column of
# R3, which is why the zero third row of Lambda_3 is admissible.
# ------------------------------------------------------------------
A3 := diff_algebra([dx, x], [dy, y], [dz, z], polynom = {x, y, z}):

P3 := Matrix([[-z*dy^2-dz^2, z*dx*dy, dx*dz, 0, 0, -1],
  [z*dx*dy, -z*dx^2-dz^2, dy*dz, 0, -1, 0],
  [dx*dz, dy*dz, -dx^2-dy^2, -1, 0, 0]]):
S3 := Matrix([[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, -1],
  [0, -1, 0],
  [-1, 0, 0]]):

row_relation_3 := GT_row_relation_search(P3, [0, 1, -1], A3):
oracle3 := GT_projectivity_proof_oracle(P3, S3, A3,
    evalb(row_relation_3["relationFound"] = false)):
assert_true("Ex3 operator-order identity P3*S3 = I", oracle3["rightInverseIdentity"]):
assert_true("Ex3 no bounded row relation", oracle3["rowInjective"]):
assert_true("Ex3 projectivity certificate", oracle3["projective"]):

printf("ALL MAPLE EXT1 CROSSCHECK TESTS PASSED\n"):
