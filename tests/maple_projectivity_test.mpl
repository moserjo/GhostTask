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

# Maple must use Ore composition, not ordinary commutative multiplication.
A := diff_algebra([dt, t], polynom = {t}):
assert_true("Weyl commutator", simplify(
    Ore_algebra:-skew_product(dt, t, A) - t*dt) = 1):

# A small split presentation with a nontrivial one-column augmentation.
R := Matrix([[1], [0]]):
LambdaSpace := [0, 1]:
witnessSpace := [0, 1, -1]:
relationSpace := [0, 1, -1]:

proof_oracle := proc(P)
    GT_bounded_projectivity_certificate(P, witnessSpace,
                                        relationSpace, A)
end proc:

nontriviality_oracle := proc(Lambda)
local c;
    c := GT_nontriviality_within_bounds(Lambda, witnessSpace, A):
    table(["nontrivial" = c["nontrivialWithinWitnessSpace"],
           "certificate" = c])
end proc:

P := GT_concat_columns(R, -Matrix([[0], [1]])):
S := Matrix([[1, 0], [0, -1]]):
exact := GT_projectivity_proof_oracle(P, S, A, true):
assert_true("exact right-inverse certificate", exact["projective"]):
assert_true("exact witness identity", exact["rightInverseIdentity"]):

bounded := GT_bounded_projectivity_certificate(P, witnessSpace,
                                               relationSpace, A):
assert_true("bounded projectivity certificate", bounded["projectiveWithinBounds"]):

bad := GT_rule_out_projectivity(
    GT_concat_columns(R, -Matrix([[1], [0]])),
    witnessSpace, relationSpace, A):
assert_true("bounded rejection detects missing split",
    bad["ruledOutWithinBounds"]):

g := GT_generate_projective_lambda(R, LambdaSpace, 1, A,
                                   proof_oracle, nontriviality_oracle):
assert_true("finite generator finds a Lambda", g["found"]):
assert_true("finite generator finds one column", g["minimumColumns"] = 1):
assert_true("finite generator has finite completeness flag",
    g["completeWithinOperatorSpace"]):

allSolutions := GT_enumerate_minimal_projective_lambdas(
    R, LambdaSpace, 1, A, proof_oracle, nontriviality_oracle):
assert_true("minimal enumeration finds solutions", allSolutions["found"]):
assert_true("minimal enumeration is complete",
    allSolutions["completeWithinOperatorSpace"]):

ball := GT_operator_ball([t], [-1, 0, 1], 1, A):
assert_true("generated Weyl ball contains t", member(t, ball)):
assert_true("generated Weyl ball contains -t", member(-t, ball)):

Dense := Matrix([[0, 1], [1, 0]]):
assert_true("row support accepts dense Lambda",
    GT_column_support(Dense)):

print("ALL MAPLE PROJECTIVITY TESTS PASSED"):
