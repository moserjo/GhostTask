-- Constructive extension-module layer for Ghost Tasking, in the paper's
-- own operator convention.
--
-- Mathematical basis (Cluzeau-Quadrat, "Serre's reduction of linear partial
-- differential systems based on holonomy", MTNS 2010, Thm 3.1; Serre;
-- Stafford): for a full-row-rank presentation R in D^(q x p), the following
-- are equivalent for the augmentation P = [R | -Lambda]:
--
--   1. the presented left module E = D^(1 x (p+r))/(D^(1 x q) P) is
--      stably free,
--   2. P admits a right inverse S with P*S = id_q in the operator order
--      (P*S)_ij = sum_k P_ik S_kj,
--   3. ext^1_D(E, D) = D^q / (P D^(p+r)) = 0,
--   4. the residue classes of the columns of Lambda generate the torsion
--      right D-module ext^1_D(M, D) = D^q / (R D^p).
--
-- Convention audit.  Macaulay2's matrix product over a Weyl algebra
-- composes entries in the opposite ring order: (A*B)_ij = sum_k B_kj A_ik.
-- Consequently a Macaulay2 identity "P*S == id" is not the operator-order
-- right-inverse identity above, and `image P` is the left span of the
-- columns, not the right span R D^p.  The two sides are exchanged by the
-- standard involution tau (x -> x, d -> -d, tau(ab) = tau(b) tau(a)),
-- applied entrywise by Dmodules' Dtransposition without reshaping:
--
--   sum_k P_ik S_kj = delta_ij
--     <=>  (Dtransposition P) * (Dtransposition S) == id  in Macaulay2.
--
-- Every routine in this file therefore converts inputs into the
-- tau-world, computes with Macaulay2's native machinery there, and
-- converts returned witnesses back.  Returned identities hold in the
-- operator order and are cross-checked externally by Maple's
-- Ore_algebra:-skew_product (tests/maple_ext1_crosscheck.mpl).
--
-- True row injectivity ker_D(.P) = { mu in D^(1 x q) : mu P = 0 } = 0 is
-- equivalent to `syz transpose P == 0` in Macaulay2 (the left row kernel
-- with coefficients acting from the left matches Macaulay2's reversed
-- product on the plain transpose; no involution is involved).

needsPackage "Dmodules";

gtTau = M -> Dtransposition M;

-- ext^1_D(M, D) = D^q/(R D^p): in the tau-world this is the Macaulay2
-- cokernel of Dtransposition R.
ext1Module = R -> coker gtTau R;

-- The extension module vanishes iff R itself admits an operator-order
-- right inverse; then the original system is already stably free and no
-- ghost column is needed.
ext1IsZero = R -> (
    Rt := gtTau R;
    isSubset(image id_(target Rt), image Rt)
);

-- Dimension of the extension module through its characteristic variety.
-- Central parameters enlarge every dimension equally, so torsion and
-- holonomicity are read against the free rank-one module and the number
-- of genuine Weyl pairs.  Holonomicity of ext^1 is the Cluzeau-Quadrat
-- sufficient condition for a one-column reduction.
ext1Dimension = R -> Ddim coker gtTau R;

gtWeylPairCount = D -> # (options D).WeylAlgebra;

ext1IsTorsion = R -> (
    D := ring R;
    Ddim coker gtTau R < Ddim (D^1)
);

ext1IsHolonomic = R -> (
    D := ring R;
    Ddim coker gtTau R === Ddim (D^1) - gtWeylPairCount D
);

-- True row injectivity of a presentation.
gtRowInjective = P -> numColumns syz transpose P == 0;

-- Columns of Lambda generate ext^1 iff [R | -Lambda] has an
-- operator-order right inverse.  The returned witness S satisfies
-- sum_k P_ik S_kj = delta_ij; the identity is verified in the tau-world
-- before returning.
lambdaGeneratesExt1 = (R, Lambda) -> (
    P := R | (-Lambda);
    Pt := gtTau P;
    I := id_(target Pt);
    W := I // Pt;
    split := W =!= null and Pt*W == I;
    new HashTable from {
        "P" => P,
        "witness" => if split then gtTau W else null,
        "generates" => split,
        "rowInjective" => gtRowInjective P,
        "criterion" =>
            "operator-order P*S = id_q iff Lambda generates D^q/(R D^p)"
    }
);

-- Relation module of a generating block: K = { v in D^r : Lambda v in
-- R D^p } with operator-order products, from the tau-world syzygies of
-- [Lambda | R].  When Lambda generates, ext^1 = D^r / K on the chosen
-- generators.  The generators of K are returned as a matrix whose
-- columns are relation vectors in the operator order.
lambdaRelationGenerators = (R, Lambda) -> (
    r := numColumns Lambda;
    relations := syz gtTau (Lambda | R);
    gtTau (relations^(toList(0..r-1)))
);

-- Strip inferred grading so matrices from different sources share a
-- common free target.
gtSameTarget = (D, q, M) -> map(D^q, , entries M);

-- Complete family theorem, effective form.
--
-- Fix a certified generating block Lambda0 with r columns and relation
-- generators K = lambdaRelationGenerators(R, Lambda0).  For any Lambda
-- in D^(q x r), operator-order products throughout:
--
--   (a) there exist U in D^(r x r), X in D^(p x r) with
--       Lambda = Lambda0 U + R X, because Lambda0's columns generate
--       D^q/(R D^p);
--   (b) Lambda generates ext^1 iff the columns of U generate D^r / K.
--
-- The full set of admissible r-column blocks is therefore
--   { Lambda0 U + R X : U in D^(r x r), X in D^(p x r),
--                       columns of U generate D^r / K },
-- with both the decomposition and the generation condition decided by
-- Groebner computations.  All three checks below are reported so that a
-- caller can assert their consistency.
lambdaFamilyMembership = (R, Lambda0, Lambda) -> (
    r := numColumns Lambda0;
    if numColumns Lambda != r then
        error "lambdaFamilyMembership compares equal column counts";
    D := ring R;
    q := numRows R;
    Gt := gtSameTarget(D, q, gtTau (Lambda0 | R));
    factorT := (gtSameTarget(D, q, gtTau Lambda)) // Gt;
    p := numColumns R;
    Ut := gtSameTarget(D, r, factorT^(toList(0..r-1)));
    Xt := gtSameTarget(D, p, factorT^(toList(r..(r + p - 1))));
    U := gtTau Ut;
    X := gtTau Xt;
    -- reassemble in the tau-world, compare entrywise
    decomposed := entries((gtTau Lambda0)*Ut + (gtTau R)*Xt) ==
        entries gtSameTarget(D, q, gtTau Lambda);
    generates := (lambdaGeneratesExt1(R, Lambda))#"generates";
    K := lambdaRelationGenerators(R, Lambda0);
    -- columns of U generate D^r/K iff [tau U | tau K] spans in the
    -- tau-world; strip gradings so all ambients agree
    UmodK := isSubset(image id_(D^r),
        image (gtSameTarget(D, r, Ut) | gtSameTarget(D, r, gtTau K)));
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
-- ext^1_D(M, D^(1 x r)) = D^(q x r)/(R D^(p x r)): tau-world reduction
-- of each column modulo the column span of tau R.  Blocks with equal
-- normal forms define the same Serre-reduction data.
lambdaNormalForm = (R, Lambda) -> gtTau ((gtTau Lambda) % (image gtTau R));

-- Certified minimum report.  The lower bound r >= 1 is exact whenever
-- ext^1 is nonzero, so a certified one-column block proves global
-- minimality.  For two columns the report states the honest status:
-- certified upper bound, with Stafford's theorem bounding every torsion
-- ext^1 by two generators.
minimumColumnReport = (R, Lambda) -> (
    r := numColumns Lambda;
    cert := lambdaGeneratesExt1(R, Lambda);
    nonzero := not ext1IsZero R;
    exact := (r == 1 and nonzero and cert#"generates");
    new HashTable from {
        "columns" => r,
        "ext1Nonzero" => nonzero,
        "generates" => cert#"generates",
        "rowInjective" => cert#"rowInjective",
        "witness" => cert#"witness",
        "minimumIsExact" => exact,
        "status" => if exact then
            "one certified column and ext^1 != 0: global minimum r = 1"
        else if cert#"generates" then
            "certified upper bound; Stafford bound gives r <= 2 for torsion ext^1"
        else "no certificate"
    }
);

-- A single ghost column is automatically nontrivial for q >= 2: a right
-- D-module surjection D -> D^q is impossible over the noetherian domain
-- D by rank over its division ring of fractions.  The explicit check is
-- retained for wider blocks, in the operator convention.
lambdaIsPIGPTrivial = Lambda -> (
    Lt := gtTau Lambda;
    isSubset(image id_(target Lt), image Lt)
);
