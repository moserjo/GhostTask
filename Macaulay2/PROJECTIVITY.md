# Projectivity rejection oracle

The double-annihilator test in `constructive_lambda.m2` checks a complete
parametrization. It does not check projectivity.

For a full-row-rank presentation

```text
E = D^(1-by-n) / D^(1-by-q) P,
```

the presentation is split exactly when there is a right inverse `S` with

```text
P*S = id_q.
```

`negativeProjectivityCertificate P` computes the exact inclusion
`image(id_q) <= image(P)` over the Weyl algebra. If that inclusion fails while
the row-injectivity check succeeds, `ruledOut` is true: this specific
presentation cannot be projective. The returned `partialFactor` and
`residual` make the failed identity factorization inspectable.

The regression test rules out the one-column candidates from all three paper
examples. This is a genuine projectivity rejection, not merely failure of the
parametrization test.

`projectivityProofOracle P` is the corresponding positive oracle. It computes
the exact factor `S = id_q // P` and accepts it only after checking
`P*S = id_q` in the original Weyl algebra, together with row injectivity.
`projectivityProofWithWitness` checks a caller-supplied explicit `S`. Thus a
positive result contains a machine-checked splitting witness, not a boolean
surjectivity heuristic.

The three paper systems have positive two-column witnesses in
`tests/projectivity_proof_test.m2`; the old one-column candidates remain
rejected by the negative test.

`finiteOperatorSpace(basis, coefficients)` and
`generateProjectiveLambda(R, operatorSpace, maxColumns)` provide constructive
generation. The generator enumerates every nonzero one-column candidate and,
when requested, every two-column candidate from the finite space, invoking the
positive proof oracle on each. A successful result contains a projectivity
proof; an unsuccessful result is marked `completeWithinSpace` and is a true
bounded no-go only for that declared finite space and column bound. There is
no identity fallback.

Run it from the repository root:

```bash
M2 --no-readline tests/projectivity_certificate_test.m2
M2 --no-readline tests/projectivity_proof_test.m2
M2 --no-readline tests/projective_lambda_generation_test.m2
```
