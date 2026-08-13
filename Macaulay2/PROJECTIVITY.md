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

Run it from the repository root:

```bash
M2 --no-readline tests/projectivity_certificate_test.m2
```
