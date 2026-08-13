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
proof and records the minimum ghost-column count. It rejects every Lambda
whose own columns already have full image, which removes the identity and all
PIGP-equivalent invertible or redundant augmentations. An unsuccessful result
is marked `completeWithinSpace` and is a true bounded no-go only for that
declared finite space and column bound. There is no identity fallback.

`enumerateMinimalProjectiveLambdas` exhausts the minimum column case and
returns every literal Lambda in that finite space, together with its checked
projectivity certificate. `solutionCount == 1` is a uniqueness certificate
relative to the exact finite space and its ordered representation. Multiple
matrices can still describe the same construction after a change of basis in
the ghost columns, so literal uniqueness is stronger than uniqueness up to
equivalence.

Run it from the repository root:

```bash
M2 --no-readline tests/projectivity_certificate_test.m2
M2 --no-readline tests/projectivity_proof_test.m2
M2 --no-readline tests/projective_lambda_generation_test.m2
```

The next layer, `operatorBall` and `generalProjectiveLambdaSearch`, removes
the need to hand-write a finite operator basis. Given Weyl-algebra generators,
a coefficient alphabet, and a word-length bound, it enumerates the resulting
finite ball and runs the exact proof oracle on every allowed one- or
two-column Lambda. The identity augmentation is explicitly excluded.

The result is complete within the declared ball. Enlarging the word and
coefficient bounds gives a semi-decision procedure: a nontrivial projective
Lambda will be found once it enters the searched balls, but a bounded failure
does not prove that no Lambda exists globally. The search tries one ghost
column before two, and the reported minimum is certified within the ball.
`generalProjectiveLambdaSolutions` returns the full literal solution set at
that minimum. This is the strongest constructive general search currently
implemented in Macaulay2; it is not a terminating direct generator from an
arbitrary Ext module.

The official [OreModules documentation](https://who.rocq.inria.fr/Alban.Quadrat/OreModules/package.html)
provides `Exti`, resolutions, syzygies, and right-inverse routines, and the
[Stafford package](https://who.rocq.inria.fr/Alban.Quadrat/OreModules/stafford.html)
documents constructive Weyl-algebra module and ideal algorithms. They are the
natural next backend for a fully general Ext-based generator, but the local
Macaulay2 installation does not provide a noncommutative `Ext` implementation
or that adapter yet.

Run the general-ball test with:

```bash
M2 --no-readline tests/general_lambda_generation_test.m2
```
