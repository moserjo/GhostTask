with(LinearAlgebra):
with(Ore_algebra):

# Maple backend for the finite GhostTask projectivity search.
#
# Maple represents Ore-algebra elements by commutative expressions. Their
# product is operator composition and must therefore use
# Ore_algebra:-skew_product. This file keeps that distinction explicit in
# every matrix product.

GT_zero_matrix := proc(m, n)
    Matrix(m, n, (i, j) -> 0)
end proc:

GT_identity_matrix := proc(n)
    Matrix(n, n, (i, j) -> piecewise(i = j, 1, 0))
end proc:

GT_ore_matrix_multiply := proc(A, B, alg)
local m, n, p, i, j, k;
    m := RowDimension(A):
    n := ColumnDimension(A):
    p := ColumnDimension(B):
    if n <> RowDimension(B) then
        error "incompatible matrix dimensions":
    end if:
    Matrix(m, p, (i, j) ->
        add(Ore_algebra:-skew_product(A[i, k], B[k, j], alg), k = 1 .. n))
end proc:

GT_ore_matrix_equal := proc(A, B)
local i, j;
    if RowDimension(A) <> RowDimension(B)
       or ColumnDimension(A) <> ColumnDimension(B) then
        return false
    end if:
    for i to RowDimension(A) do
        for j to ColumnDimension(A) do
            if simplify(expand(A[i, j] - B[i, j])) <> 0 then
                return false
            end if:
        end do:
    end do:
    true
end proc:

GT_nonzero_vector := proc(v)
local x;
    for x in v do
        if simplify(x) <> 0 then
            return true
        end if:
    end do:
    false
end proc:

GT_all_vectors := proc(values, q)
local tails, out, v, x;
    if q = 0 then
        return [[]]
    end if:
    tails := GT_all_vectors(values, q - 1):
    out := []:
    for x in values do
        for v in tails do
            out := [op(out), [x, op(v)]]:
        end do:
    end do:
    out
end proc:

GT_column_matrix := proc(v)
    Matrix(nops(v), 1, (i, j) -> v[i])
end proc:

GT_concat_columns := proc(A, B)
local m, a, b;
    m := RowDimension(A):
    a := ColumnDimension(A):
    b := ColumnDimension(B):
    if m <> RowDimension(B) then
        error "incompatible column blocks":
    end if:
    Matrix(m, a + b,
        (i, j) -> piecewise(j <= a, A[i, j], B[i, j - a]))
end proc:

GT_zero_column := proc(A)
local i;
    for i to RowDimension(A) do
        if simplify(A[i, 1]) <> 0 then
            return false
        end if:
    end do:
    true
end proc:

GT_column_support := proc(Lambda)
local i, j, nonzero;
    for i to RowDimension(Lambda) do
        nonzero := false:
        for j to ColumnDimension(Lambda) do
            if simplify(Lambda[i, j]) <> 0 then
                nonzero := true:
            end if:
        end do:
        if not nonzero then
            return false
        end if:
    end do:
    true
end proc:

GT_append_unique := proc(out, x)
local y;
    for y in out do
        if simplify(expand(x - y)) = 0 then
            return out
        end if:
    end do:
    [op(out), x]
end proc:

GT_finite_operator_space := proc(basis, coefficients)
local out, b, c, x;
    out := []:
    for b in basis do
        for c in coefficients do
            x := expand(c*b):
            out := GT_append_unique(out, x):
        end do:
    end do:
    out
end proc:

GT_operator_ball := proc(generators, coefficients, degree, alg)
local out, frontier, next_frontier, g, w, c, x, d;
    out := GT_finite_operator_space([1], coefficients):
    frontier := [1]:
    for d from 1 to degree do
        next_frontier := []:
        for w in frontier do
            for g in generators do
                x := Ore_algebra:-skew_product(w, g, alg):
                next_frontier := GT_append_unique(next_frontier, x):
            end do:
        end do:
        for c in coefficients do
            for x in next_frontier do
                out := GT_append_unique(out, expand(c*x)):
            end do:
        end do:
        frontier := next_frontier:
    end do:
    out
end proc:

# Exact certificate for a supplied right inverse. The row-injectivity flag
# must come from an independent syzygy calculation. The procedure never
# turns a bounded search into a global theorem.
GT_projectivity_proof_oracle := proc(P, S, alg, row_injective_certificate)
local identity_matrix, exact, answer;
    identity_matrix := GT_identity_matrix(RowDimension(P)):
    exact := GT_ore_matrix_equal(
        GT_ore_matrix_multiply(P, S, alg), identity_matrix):
    answer := table():
    answer["witness"] := S:
    answer["rightInverseIdentity"] := exact:
    answer["rowInjective"] := row_injective_certificate:
    answer["projective"] := exact and row_injective_certificate:
    answer["scope"] := "exact supplied witness plus supplied row-injectivity certificate":
    answer
end proc:

GT_right_inverse_search := proc(P, witness_space, alg)
local n, q, vectors, out, i, j, candidate, S;
    n := ColumnDimension(P):
    q := RowDimension(P):
    vectors := GT_all_vectors(witness_space, n*q):
    out := table():
    out["found"] := false:
    out["completeWithinWitnessSpace"] := true:
    out["tested"] := 0:
    for candidate in vectors do
        S := Matrix(n, q, (i, j) -> candidate[(i - 1)*q + j]):
        out["tested"] := out["tested"] + 1:
        if GT_ore_matrix_equal(
            GT_ore_matrix_multiply(P, S, alg),
            GT_identity_matrix(q)) then
            out["found"] := true:
            out["witness"] := S:
            return out
        end if:
    end do:
    out
end proc:

# Bounded analogue of the transposed-syzygy check. The finite scope is
# returned explicitly.
GT_row_relation_search := proc(P, relation_space, alg)
local q, n, vectors, out, c, i, j, value, annihilates;
    q := RowDimension(P):
    n := ColumnDimension(P):
    vectors := GT_all_vectors(relation_space, q):
    out := table():
    out["relationFound"] := false:
    out["completeWithinRelationSpace"] := true:
    out["tested"] := 0:
    for c in vectors do
        if not GT_nonzero_vector(c) then
            next
        end if:
        out["tested"] := out["tested"] + 1:
        annihilates := true:
        for j to n do
            value := add(Ore_algebra:-skew_product(c[i], P[i, j], alg),
                         i = 1 .. q):
            if simplify(expand(value)) <> 0 then
                annihilates := false:
                break
            end if:
        end do:
        if annihilates then
            out["relationFound"] := true:
            out["relation"] := c:
            return out
        end if:
    end do:
    out
end proc:

GT_bounded_projectivity_certificate := proc(P, witness_space,
                                             relation_space, alg)
local inverse, relations, answer;
    inverse := GT_right_inverse_search(P, witness_space, alg):
    relations := GT_row_relation_search(P, relation_space, alg):
    answer := table():
    answer["rightInverseFound"] := inverse["found"]:
    if inverse["found"] then
        answer["rightInverse"] := inverse["witness"]:
    else
        answer["rightInverse"] := false:
    end if:
    answer["rowInjectiveWithinRelationSpace"] :=
        not relations["relationFound"]:
    answer["projectiveWithinBounds"] :=
        inverse["found"] and not relations["relationFound"]:
    answer["completeWithinWitnessSpace"] :=
        inverse["completeWithinWitnessSpace"]:
    answer["completeWithinRelationSpace"] :=
        relations["completeWithinRelationSpace"]:
    answer["scope"] := "finite witness and relation spaces":
    answer
end proc:

GT_rule_out_projectivity := proc(P, witness_space, relation_space, alg)
local c, answer;
    c := GT_bounded_projectivity_certificate(P, witness_space,
                                             relation_space, alg):
    answer := table():
    answer["ruledOutWithinBounds"] :=
        not c["rowInjectiveWithinRelationSpace"]
        or (c["rowInjectiveWithinRelationSpace"]
            and not c["rightInverseFound"]):
    answer["certificate"] := c:
    answer["scope"] := "bounded rejection only":
    answer
end proc:

# Nontriviality is checked by looking for a right inverse of Lambda itself.
# It is bounded when the finite search is used.
GT_nontriviality_within_bounds := proc(Lambda, witness_space, alg)
local c, answer;
    c := GT_right_inverse_search(Lambda, witness_space, alg):
    answer := table():
    answer["columnImageFullWithinWitnessSpace"] := c["found"]:
    answer["nontrivialWithinWitnessSpace"] := not c["found"]:
    answer["certificate"] := c:
    answer
end proc:

GT_candidate_list := proc(R, operator_space, columns)
local q, vectors, out, i, j, L;
    q := RowDimension(R):
    vectors := GT_all_vectors(operator_space, q):
    out := []:
    if columns = 1 then
        for i to nops(vectors) do
            if GT_nonzero_vector(vectors[i]) then
                out := [op(out), GT_column_matrix(vectors[i])]:
            end if:
        end do:
        return out
    end if:
    for i from 1 to nops(vectors) - 1 do
        if GT_nonzero_vector(vectors[i]) then
            for j from i + 1 to nops(vectors) do
                if GT_nonzero_vector(vectors[j]) then
                    L := GT_concat_columns(GT_column_matrix(vectors[i]),
                                            GT_column_matrix(vectors[j])):
                    out := [op(out), L]:
                end if:
            end do:
        end if:
    end do:
    out
end proc:

# proof_oracle(P) and nontriviality_oracle(Lambda) are callbacks. This keeps
# candidate generation independent of the chosen exact certificate engine.
GT_generate_projective_lambda := proc(R, operator_space, max_columns,
                                       alg, proof_oracle,
                                       nontriviality_oracle)
local k, candidates, Lambda, P, proof, nontrivial, answer, tested;
    if max_columns < 1 or max_columns > 2 then
        error "the GhostTask bound supports one or two columns":
    end if:
    tested := 0:
    for k from 1 to max_columns do
        candidates := GT_candidate_list(R, operator_space, k):
        for Lambda in candidates do
            tested := tested + 1:
            nontrivial := nontriviality_oracle(Lambda):
            if not nontrivial["nontrivial"] then
                next
            end if:
            P := GT_concat_columns(R, -Lambda):
            proof := proof_oracle(P):
            if proof["projective"] then
                answer := table():
                answer["found"] := true:
                answer["Lambda"] := Lambda:
                answer["P"] := P:
                answer["proof"] := proof:
                answer["minimumColumns"] := k:
                answer["tested"] := tested:
                answer["completeWithinOperatorSpace"] := true:
                answer["scope"] := "finite operator space and supplied proof oracle":
                return answer
            end if:
        end do:
    end do:
    answer := table():
    answer["found"] := false:
    answer["tested"] := tested:
    answer["completeWithinOperatorSpace"] := true:
    answer["scope"] := "finite operator space and supplied proof oracle":
    answer
end proc:

GT_enumerate_minimal_projective_lambdas := proc(R, operator_space,
                                                 max_columns, alg,
                                                 proof_oracle,
                                                 nontriviality_oracle)
local first, k, candidates, Lambda, P, proof, nontrivial, solutions, answer;
    first := GT_generate_projective_lambda(R, operator_space, max_columns,
                                           alg, proof_oracle,
                                           nontriviality_oracle):
    if not first["found"] then
        return first
    end if:
    k := first["minimumColumns"]:
    candidates := GT_candidate_list(R, operator_space, k):
    solutions := []:
    for Lambda in candidates do
        nontrivial := nontriviality_oracle(Lambda):
        if not nontrivial["nontrivial"] then
            next
        end if:
        P := GT_concat_columns(R, -Lambda):
        proof := proof_oracle(P):
        if proof["projective"] then
            solutions := [op(solutions), Lambda]:
        end if:
    end do:
    answer := table():
    answer["found"] := true:
    answer["minimumColumns"] := k:
    answer["solutions"] := solutions:
    answer["solutionCount"] := nops(solutions):
    answer["uniqueWithinOperatorSpace"] := nops(solutions) = 1:
    answer["completeWithinOperatorSpace"] := true:
    answer
end proc:
