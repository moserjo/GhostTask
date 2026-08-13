-- Constructive extension-module layer for Ghost Tasking.
--
-- Mathematical basis (Cluzeau-Quadrat, "Serre's reduction of linear partial
-- differential systems based on holonomy", MTNS 2010, Thm 3.1; Serre;
-- Stafford): for a full-row-rank presentation R in D^(q x p), the following
-- are equivalent for an augmentation P = [R | -Lambda]:
--
--   1. the presented module E is stably free,
--   2. P admits a right inverse S with P*S = id_q,
--   3. ext^1_D(E, D) = D^q / (P D^(p+r)) = 0,
--   4. the residue classes of the columns of Lambda generate the torsion
--      right D-module ext^1_D(M, D) = D^q / (R D^p).
--
-- The right-inverse certificate checked by projectivity_certificate.m2 is
-- therefore exactly the statement that Lambda's columns generate the
-- extension module.  This file adds the module-level layer: the extension
-- module itself, its dimension, the relation module of a certified Lambda,
-- and the complete parameterization of all admissible Lambda blocks at a
-- fixed column count.
--
-- Convention.  Macaulay2 matrices over a Weyl algebra act on columns with
-- right-hand multipliers, so `image P` is the right-submodule R D^p spanned
-- by the columns and `coker R` is the right D-module D^q/(R D^p), which is
-- the extension module above.  Row-injectivity certificates use the
-- transposed convention, exactly as in projectivity_certificate.m2.

needsPackage "Dmodules";

-- ext^1_D(M, D) = D^q/(R D^p) in the column convention described above.
ext1Module = R -> coker R;

-- The extension module vanishes iff R itself already admits a right
-- inverse.  In that case the original system is already stably free and no
-- ghost column is needed.
ext1IsZero = R -> (
    isSubset(image id_(target R), image R)
);

-- Dimension of the extension module in the sense of the characteristic
-- variety.  Central parameters enlarge every dimension by the same
-- amount, so both the torsion and the holonomicity test compare against
-- the dimension of the free rank-one module and the number of genuine
-- Weyl pairs.  Holonomicity of ext^1 is the Cluzeau-Quadrat sufficient
-- condition for a one-column reduction (Serre's reduction based on
-- holonomy, MTNS 2010, Thm 4.1).
ext1Dimension = R -> Ddim coker R;

gtWeylPairCount = D -> # (options D).WeylAlgebra;

ext1IsTorsion = R -> (
    D := ring R;
    Ddim coker R < Ddim (D^1)
);

ext1IsHolonomic = R -> (
    D := ring R;
    Ddim coker R === Ddim (D^1) - gtWeylPairCount D
);

-- Columns of Lambda generate ext^1 iff [R | -Lambda] has a right inverse.
-- The returned certificate contains the checked witness.
lambdaGeneratesExt1 = (R, Lambda) -> (
    P := R | (-Lambda);
    I := id_(target P);
    witness := I // P;
    split := witness =!= null and P*witness == I;
    new HashTable from {
        "P" => P,
        "witness" => witness,
        "generates" => split,
        "criterion" =>
            "P*S = id_q iff Lambda's columns generate D^q/(R D^p)"
    }
);

-- Relation module of a generating block: K = { v in D^r : Lambda v lies in
-- R D^p }, computed from the syzygies of [Lambda | R].  When Lambda
-- generates, ext^1 = D^r / K, so K is the presentation of the extension
-- module on the chosen generators.
lambdaRelationModule = (R, Lambda) -> (
    r := numColumns Lambda;
    relations := syz(Lambda | R);
    image (relations^(toList(0..r-1)))
);

-- Strip the inferred grading so that matrices assembled from different
-- sources live over one common free target.
gtSameTarget = (D, q, M) -> map(D^q, , entries M);

-- Complete family theorem, in effective form.
--
-- Fix a certified generating block Lambda0 with r columns and relation
-- module K = lambdaRelationModule(R, Lambda0).  Then for any Lambda in
-- D^(q x r):
--
--   (a) there exist U in D^(r x r) and X in D^(p x r) with
--       Lambda = Lambda0*U + R*X, because the columns of Lambda0 generate
--       D^q/(R D^p);
--   (b) Lambda generates ext^1 iff the columns of U generate D^r / K.
--
-- Consequently the full set of admissible r-column blocks is
--   { Lambda0*U + R*X : X in D^(p x r), U in D^(r x r),
--                       columns of U generate D^r/K },
-- and both the decomposition (U, X) and the generation condition are
-- effective Groebner computations.  This routine returns the decomposition
-- for a supplied Lambda together with both checks.
lambdaFamilyMembership = (R, Lambda0, Lambda) -> (
    r := numColumns Lambda0;
    if numColumns Lambda != r then
        error "lambdaFamilyMembership compares equal column counts";
    D := ring R;
    q := numRows R;
    G := gtSameTarget(D, q, Lambda0 | R);
    factor := (gtSameTarget(D, q, Lambda)) // G;
    U := factor^(toList(0..r-1));
    X := factor^(toList(r..(r + numColumns R - 1)));
    -- entry comparison: matrix == also compares inferred grading metadata
    decomposed := entries(Lambda0*U + R*X) == entries Lambda;
    generates := (lambdaGeneratesExt1(R, Lambda))#"generates";
    K := lambdaRelationModule(R, Lambda0);
    -- columns of U generate D^r/K iff [U | K-generators] spans D^r
    UmodK := isSubset(image id_(target U), image U + K);
    new HashTable from {
        "U" => U,
        "X" => X,
        "decomposed" => decomposed,
        "lambdaGenerates" => generates,
        "UGeneratesModK" => UmodK,
        "consistent" => decomposed and (generates === UmodK),
        "statement" =>
            "Lambda = Lambda0*U + R*X; admissible iff U generates D^r/K"
    }
);

-- Canonical representative of the residue class of Lambda in
-- ext^1_D(M, D^(1 x r)) = D^(q x r)/(R D^(p x r)): reduction of each
-- column modulo a Groebner basis of the column module of R.  Two blocks
-- with equal normal forms define the same Serre-reduction data
-- (Cluzeau-Quadrat, Thm 3.1, final statement).
lambdaNormalForm = (R, Lambda) -> Lambda % (gb image R);

-- Certified minimum. The lower bound r >= 1 is exact whenever the
-- extension module is nonzero; a certified one-column block therefore
-- proves minimality outright.  For r = 2 the routine reports the honest
-- status: minimal relative to the searched spaces, globally bounded by
-- Stafford's two-generator theorem for torsion modules.
minimumColumnReport = (R, Lambda) -> (
    r := numColumns Lambda;
    cert := lambdaGeneratesExt1(R, Lambda);
    nonzero := not ext1IsZero R;
    exact := (r == 1 and nonzero and cert#"generates");
    new HashTable from {
        "columns" => r,
        "ext1Nonzero" => nonzero,
        "generates" => cert#"generates",
        "witness" => cert#"witness",
        "minimumIsExact" => exact,
        "status" => if exact then
            "one certified column and ext^1 != 0: global minimum r = 1"
        else if cert#"generates" then
            "certified upper bound; Stafford bound gives r <= 2 for torsion ext^1"
        else "no certificate"
    }
);

-- Nontriviality of a single ghost column is automatic for q >= 2: a
-- surjection D -> D^q would make D^q cyclic, which is impossible over the
-- noetherian domain D (rank over the division ring of fractions).  The
-- check is retained for rectangular blocks.
lambdaIsPIGPTrivial = Lambda -> (
    isSubset(image id_(target Lambda), image Lambda)
);
