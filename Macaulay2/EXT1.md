# Constructive extension-module layer

`ext1_constructive.m2` turns the finite Lambda search of the previous
pull request into a theory-level pipeline for the ghost-tasking
construction.  The mathematical frame is Serre's reduction (Boudellioua
and Quadrat, Math. Comput. Sci. 4, 2010; Cluzeau and Quadrat,
J. Symb. Comput. 47, 2012): for a full-row-rank presentation `R` in
`D^(q x p)`, the augmentation `P = [R | -Lambda]` presents a stably free
module exactly when `P` has an operator-order right inverse, exactly
when the columns of `Lambda` generate the torsion right module
`ext^1_D(M, D) = D^q/(R D^p)`.  The minimum number of ghost columns is
the minimal number of generators of that module.

## Convention audit

Macaulay2 multiplies matrix entries over a Weyl algebra in the opposite
ring order: `(A*B)_ij = sum_k B_kj A_ik`.  A Macaulay2 identity
`P*S == id` is therefore not the operator identity `sum_k P_ik S_kj =
delta_ij`; the two are exchanged by the standard involution
(`Dtransposition`, entrywise, no reshaping).  The earlier
`projectivity_certificate.m2` layer certified the mirror statement.
Every function in `ext1_constructive.m2` converts to the involuted
presentation, computes there, converts witnesses back, and asserts
operator-order identities.  `tests/maple_ext1_crosscheck.mpl` verifies
the returned witnesses independently in Maple through
`Ore_algebra:-skew_product`, which composes in the written order.

True row injectivity `ker_D(.P) = 0` is `syz transpose P == 0`; the
extension module is `coker Dtransposition R`.

## What is certified for the three paper systems

- Experiment 1 (shear transport, Weyl algebra in `t, x`, parameter `a`):
  `ext^1` is torsion of dimension 4, not holonomic, and cyclic.
  `Lambda = (-1, -t)` is a certified generator, so the minimum is
  exactly one ghost column.  The eight literal solutions of the finite
  search are all members of the single parameterized family.
- Experiment 2 (tripendulum): the minimum is ring-dependent.  Over the
  Weyl algebra with coefficient field `QQ(l, g)` the extension module is
  holonomic, hence cyclic by a theorem, and `Lambda = (-1, -t, 0)` and
  the paper's original guess `(-1, -dt, -t)` are certified generators:
  minimum one.  Over the polynomial ring `QQ[l, g]` those columns fail
  (witnesses need `1/g`) and the two-column block stays certified.
- Experiment 3 (anisotropic magnetostatics, reluctivity `diag(1,1,z)`):
  `ext^1` is torsion of dimension 5, not holonomic, and nevertheless
  cyclic: `Lambda = (1, xz, 0)` is a certified generator, so the
  minimum is exactly one ghost column.  This supersedes the two-column
  minimum previously reported from the small finite ansatz.

## Complete parameterized families

For a certified generating block `Lambda0` with `r` columns and
relation module `K = lambdaRelationGenerators(R, Lambda0)`, the full
set of admissible `r`-column blocks is

    { Lambda0 U + R X : U in D^(r x r), X in D^(p x r),
                        columns of U generate D^r / K }.

Membership is effective: `lambdaFamilyMembership` computes `(U, X)` by
Groebner division and decides the generation condition.  For one
column, nontriviality is automatic (a single column cannot generate
`D^q` for `q >= 2`), so the PIGP shortcut is excluded by rank alone.

## Status of the general theory

Stafford's two-generator theorem (J. London Math. Soc. 18, 1978,
Thm 3.7) bounds every torsion `ext^1` by two generators, and the bound
is effective (Hillebrand and Schmale, J. Symb. Comput. 32, 2001;
Leykin, J. Symb. Comput. 38, 2004).  One column is guaranteed whenever
`ext^1` is holonomic (Cluzeau and Quadrat, 2012).  Whether every
finitely generated torsion module over the Weyl algebra is cyclic is
Stafford's conjecture and remains open (Bellamy, J. London Math. Soc.,
2026, survey).  The three example systems are decided by terminating
Groebner certificates; no unbounded search is part of any claim.

## Tests

    M2 --no-readline tests/ext1_constructive_test.m2
    maple -q tests/maple_ext1_crosscheck.mpl   # from tests/, Maple 2023
