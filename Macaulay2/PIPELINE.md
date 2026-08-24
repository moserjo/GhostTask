# The ghost-column pipeline, written down and made executable

This note records how the ghost columns and kernels in this repository
are actually computed, adds a second constructor for the ghost block,
and turns the hand-transcription step into a test.  Everything asserted
below is executed by `tests/ghosttask_pipeline_test.m2`; where something
was not executed, it says so.

## The pipeline as it stands

For each experiment the chain is the same.

1. Write the operator `L` of the physics by hand, in Maple, inside an
   Ore algebra built with `DefineOreAlgebra`.  Experiment 3 does this as
   `curl`, `nu`, and `Mult(curl, nu, curl, Alg)`; Experiments 1 and 2
   write the matrix out directly.
2. Append columns: first the source column, which turns an inhomogeneous
   problem into a homogeneous one, then the ghost column(s).  The result
   is the augmented operator, called `R_GT` in the worksheets and
   `RGT` in the Macaulay2 transcripts.
3. Compute the parametrisation as `Exti(Involution(R_GT, Alg), Alg, 1)`
   and take the third entry of the result; or, equivalently, in
   Macaulay2 as `Dtransposition mingens image syz Dtransposition RGT`.
4. Check by eye that `transpose B * transpose RGT` is zero and that
   `transpose mingens image syz transpose B` gives back `RGT` up to
   left combinations.
5. Copy the resulting entries by hand into `writing_kernel.py` as a
   sympy matrix, in the PCGP variable names (`D[0..2]` for the
   derivations, `x[i]` for the coordinates), where a few extra rows are
   often appended for derived observables.  `PCGP_Builder` then turns
   that matrix into the multi-task GP kernel.
6. Everything downstream -- `main_*.py`, the grid evaluations, the
   `Ex*_metrics*.ipynb` notebooks and their `Ex*_metrics_table*.py`
   companions -- consumes the generated kernel and never re-derives it.

Steps 4 and 5 are where the pipeline is fragile.  The check in step 4 is
a visual comparison of two printed matrices, and step 5 is a manual
retyping across two computer algebra conventions and two variable
namings.  Neither is covered by a test, and both are silent on failure:
a wrong kernel still trains, still fits, and simply encodes the wrong
physics.  `tests/ghosttask_pipeline_test.m2` closes that gap for the
three kernels currently in use.

### What the three systems are, concretely

Experiment 1.  `R = [[x dx + dt, a dx], [0, x dx + dt]]`, ghost column
`(-1, -dt)`, which is the `RGT3` branch of
`Experiment1_Pedagogical/Macaulay2_Pedagogical.txt`.  The kernel in
`writing_kernel.py` is exactly the `BGT3` printed there.

Experiment 2.  `R` is the three-pendulum matrix with the shared forcing
column, ghost column `(l, l dt, l t)`.  The kernel in
`writing_kernel.py` is exactly the `BGT` printed in
`Experiment2_Tripendulum/Macaulay2_Tripendulum.txt`.

Experiment 3.  This one is not what the committed Macaulay2 transcript
says, and the transcript's own filename admits it
(`Macaulay2_Magnetostatics_needsupdate.txt`).  The live system is in
`OreModules_Magnetostatics_correct.mw`, and reading the worksheet back
gives

    curl := [[0,-dz,dy],[dz,0,-dx],[-dy,dx,0]]
    v    := nu * [[1,0,0],[0,1,0],[0,0,z]]
    R    := Mult(curl, v, curl, Alg)
    R1   := R with a source column appended
    parametrizable := Exti(Involution(R_GT, Alg), Alg, 1)
    B       := parametrizable[3]
    A_part  := B[1..3]
    H       := Mult(v, curl, A_part, Alg)
    full_B  := [B[1..5]; H]

Two things follow from that and are confirmed by computation.

First, the reluctivity is `nu0 * diag(1, 1, z)`, not `diag(nu0, nu0, z)`.
The two agree at `nu0 = 1` and differ everywhere else, and `nu0` is the
parameter the inverse runs fit.  The eight-row kernel in
`writing_kernel.py` is reproduced by the first form and not by the
second.  `methods.tex` states the second form, as does the stale
Macaulay2 transcript.  The code is self-consistent; the manuscript text
is what needs correcting.

Second, the source and ghost columns of the live system are `-e1` and
`-e2`.  That is forced: rows 4 and 5 of the kernel are the first two
components of `L A`, the third component vanishes identically, and the
only pair of columns for which `R_GT B = 0` then holds is `(-e1, -e2)`.
This matches the task list in `methods.tex` -- three components of `A`,
the source `J_x` in equation one, the gauge in equation two, then the
three components of `B` -- and it means the last three kernel rows are
`curl` of the first three and carry no extra latent process.  The kernel
has two columns, so Experiment 3 runs two latent processes;
`methods.tex` calls it single-latent-process, which is a third thing to
fix in the text.

`L = curl nu curl` on its own is *not* full row rank, because
`div curl = 0` is a left relation.  Full row rank is restored by the
source column.  Any statement below about `ext^1` for Experiment 3 is
therefore a statement about `R1 = [L | -e1]` and never about `L`.

## What was verified by execution

`tests/ghosttask_pipeline_test.m2` asserts, and passes:

- each of the three `writing_kernel.py` kernels annihilates its own
  augmented operator in the operator order, entry by entry;
- each is a *complete* parametrisation, i.e. its columns span the whole
  syzygy module and not a submodule of it.  This is the decided form of
  the `R' = ...` line the transcripts check by eye;
- for Experiment 3, rows 6 to 8 are `curl` of rows 1 to 3, rows 4 and 5
  are `(L A)_1` and `(L A)_2`, and `(L A)_3` is identically zero;
- the same Experiment 3 identity fails for `nu = diag(nu0, nu0, z)`;
- `L` is not full row rank and `[L | -e1]` is.

So the three kernels in use are correct, and the hand transcription
introduced no error.  That is a check of the existing results, not a
change to them.

## A second constructor for the ghost block

`Macaulay2/stafford_reduce.m2` reduces the `q` standard generators of

    ext^1_D(M, D) = D^q / (R D^p)

to two, directly over the Weyl algebra in Macaulay2, and certifies every
intermediate block with the unmodified oracle
`lambdaGeneratesExt1Decision` from `Macaulay2/ext1_constructive.m2`.
It is a search, so an exhausted pool is a bound and not a theorem; every
positive outcome is a terminating Groebner decision and is a theorem for
the displayed `R` and the displayed block.

Applied to the three systems it returns, each certified:

| system | status | block | cpu |
| --- | --- | --- | --- |
| Experiment 1 | trivial, `q = 2` | `[e1, e2]` | 0.01 s |
| Experiment 2 over `QQ[l,g]` | reduced | `[e2, e3]` | 0.10 s |
| Experiment 2 over `QQ(l,g)` | reduced | `[e2, e3]` | 0.02 s |
| Experiment 3 over `QQ(nu0)` | reduced | `[e2, e3]` | 0.04 s |

These are cheap because the systems are small.  The reason to have the
constructor here is not speed on these four rows: it is that the ghost
block is now produced by an algorithm with a certificate attached,
rather than guessed and then inspected, and that the certificate is the
same oracle in both cases.  On larger inputs elsewhere the same
implementation is flat at about 1.3 s where a correctly-driven Maple
`StaffordReduction` took up to 103 s, and it settles three cases that
had previously been recorded as cost boundaries.  The structural reason
is that it never recomputes the torsion of `ext^1`: torsion follows from
full row rank, and the verifier decides it independently.  Those timings
were measured in another repository, on other inputs, and are not
reproduced by anything in this pull request.

## Stably free is not the same as parametrisable

`ext^1_D(M,D) = D^q/(R D^p)` and the augmentation `P = [R | -Lambda]`
are related by the Serre-reduction equivalence used in
`Macaulay2/ext1_constructive.m2`: `P` presents a *stably free* module
exactly when the columns of `Lambda` generate `ext^1`.  What the GP
actually needs is weaker -- a complete parametrisation, that is
torsion-freeness -- and the two must not be conflated.  Checking which
holds for the columns in use gives:

| system | ghost column in use | complete parametrisation | generates `ext^1` |
| --- | --- | --- | --- |
| Experiment 1 | `(-1, -dt)` | yes | **no** |
| Experiment 2, `QQ(l,g)` | `(l, l dt, l t)` | yes | yes |
| Experiment 2, `QQ[l,g]` | same column | yes | no; the witnesses need `1/l` |
| Experiment 3 | `-e2` (the gauge) | yes | **no** |

Nothing here contradicts any result recorded in this repository.  Every
kernel is a correct complete parametrisation and every printed Macaulay2
output reproduces.  What it does mean is that the Experiment 1 and
Experiment 3 augmentations are torsion-free but not stably free, so
their parametrisations have no left inverse and the latent process is
not recoverable from the tasks.  Whether that matters for the
conditioning claim behind Figure 8 is an open question this pull request
does not answer.

For Experiment 3 there is a certified alternative.  Over `QQ(nu0)`,

    Lambda = (0, x z, 1)

generates `ext^1` for `R1 = [L | -e1]`, so `[L | -e1 | -Lambda]` is
stably free; `(0,1,0)`, `(0,0,1)` and `(1, x z, 0)` are all rejected, so
the certificate discriminates.  Only one ghost column is needed, the
same width as the gauge column in use.  The corresponding explicit
parametrisation was **not** computed: the syzygy did not finish in 500 s
over `QQ(nu0)`, and no kernel is proposed for it here.  Until that is
done this is a statement about the module, not a drop-in replacement for
`writing_kernel.py`.

Note that `(1, x z, 0)`, which an earlier pull request certified for
Experiment 3, is a generator for the *stale* system of
`Macaulay2_Magnetostatics_needsupdate.txt` -- source column `(0,0,-1)`,
`nu0 = 1` -- and is rejected on the live system.  The `x z` shape
survives the correction; the position of the entries does not.

## Four things worth stating plainly

**Call `StaffordReduction` with its option.**  The Quadrat--Robertz
`StaffordReduction` initialises its internal `redgen` flag to `false`
and skips the generator-reducing step -- step 9 of the authors'
Algorithm 3 -- unless it is called as
`StaffordReduction(P, A, "reduce_generators"=true)`.  Called without
options it can return a presentation whose generator count never drops
while the relation block grows, and it reports nothing when it declines.
This repository does not call `StaffordReduction` anywhere, so no result
here is affected; the note is here so that it stays that way if the
Maple route is ever extended.

**The Quadrat--Robertz package is sound.**  Text elsewhere once
described the released package as defective.  That was wrong: it was a
call site omitting the option above.  The claim is withdrawn and their
implementation is sound.

**The theorem being used.**  For `k` of characteristic zero,
`D = A_n(k)`, and `R` of full row rank, `ext^1_D(M,D)` is a finitely
generated torsion module and therefore two-generated, and the reduction
is computable over `QQ`.  Stafford, J. London Math. Soc. (2) 18 (1978),
Theorem 3.7, p. 438; effective version in Quadrat and Robertz, Acta
Appl. Math. 133 (2014), 187--234.  Full row rank is the only hypothesis
doing work, which is exactly why the Experiment 3 source column matters.

**One column is open.**  Two generators is a theorem from 1978.  Whether
one always suffices -- Stafford's cyclicity conjecture, 3.8 -- is open,
and nothing in this pull request touches it.  The single columns
certified above are decisions about individual systems by terminating
Groebner computations, not evidence in either direction.

## Computing from a supplied `Lambda`

`Macaulay2/minimal_parametrization.m2` exposes
`gtMinimalParametrization(R, Lambda)`.  It forms `P = R | (-Lambda)`, computes
the transposed Weyl syzygies, converts them back to the operator convention,
and then checks both `P*Q = 0` and equality with the complete syzygy module.
The result reports the generic rank `numColumns(P) - numRows(P)` and only marks
the column count as certified minimum when it reaches that rank.  This guard
matters: a projectivity certificate does not by itself prove that a rank-sized
parametrisation has been found, and the Maple routine can otherwise return an
under-ranked selection.

For a hard input, `gtBoundedParametrizationFromLambda(R, Lambda, limit)` is a
diagnostic only.  It returns a bounded syzygy sample with `complete => false`,
so a `SyzygyLimit` cannot be mistaken for a parametrisation proof.  The Ex3
projective candidate `(0, x*z, 1)^T` is certified by
`lambdaGeneratesExt1Decision`, but its unrestricted syzygy remains the known
cost boundary; the regression test records that distinction explicitly.

Run the API regression with:

    M2 --script tests/minimal_parametrization_test.m2

## Running it

    M2 --script tests/ghosttask_pipeline_test.m2

About two seconds.  Nothing else in the repository is modified, no
notebook is re-executed, and no continuous-integration configuration is
added.
