# Maple projectivity backend

ghosttask_projectivity.mpl is the Maple-side finite search and certificate
layer for the GhostTask projectivity work.

Maple's Ore_algebra package represents Weyl and Ore operators by ordinary
Maple expressions. Their sum is ordinary addition, while operator composition
must use Ore_algebra:-skew_product. The source therefore performs every
matrix product through GT_ore_matrix_multiply.

The backend exposes four layers:

1. GT_rule_out_projectivity checks row relations and right inverses
   exhaustively inside declared finite witness and relation spaces. Its
   result is explicitly a bounded rejection certificate.
2. GT_projectivity_proof_oracle verifies an explicitly supplied witness
   P*S = I in the original Ore algebra. The caller must also supply an
   independent row-injectivity certificate. The routine never treats a
   finite search failure as a global theorem.
3. GT_generate_projective_lambda enumerates one-column candidates before
   two-column candidates in a finite operator space. It takes certificate
   callbacks so that the same generator can use an exact Groebner-module
   backend or a bounded witness backend.
4. GT_enumerate_minimal_projective_lambdas returns every literal solution at
   the first successful column count in the declared finite space.

The nontriviality callback checks whether the ghost block itself has a right
inverse. This excludes the identity and PIGP-equivalent full-image blocks.
The row-support predicate GT_column_support is separate from projectivity.

Run from this directory after Maple 2023 is available:

    maple -q tests/maple_projectivity_test.mpl

The current test uses a small split presentation and the Weyl commutator
D*t - t*D = 1. It is an independent behavioral test of noncommutative
matrix multiplication, exact witness checking, bounded rejection, bounded
generation, minimum-column ordering, full finite enumeration, Weyl-ball
generation, and row support.

Maple's official documentation describes the relevant primitives:

- Ore algebra:
  https://www.maplesoft.com/support/help/Maple/view.aspx?path=Ore_algebra
- Weyl algebras:
  https://www.maplesoft.com/support/help/Maple/view.aspx?path=Ore_algebra%2FWeyl_algebra
- Groebner bases for modules and skew algebras:
  https://www.maplesoft.com/support/help/Maple/view.aspx?path=Groebner%2FBasis_details

The remaining integration step is to connect Maple's left-module Groebner
output to a fully automated row-injectivity and right-inverse solver for an
arbitrary presented system. Until that adapter is written, the Maple layer
is exact for supplied witnesses and complete only inside its declared finite
search spaces. The Macaulay2 implementation remains the independently tested
certificate backend for the current PR examples.
