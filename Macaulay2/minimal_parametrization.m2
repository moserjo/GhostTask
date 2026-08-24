-- Parametrisation from a supplied GhostTask Lambda.
--
-- This is deliberately a checked wrapper around Macaulay2's syzygy
-- computation.  The Maple/OreModules MinimalParametrization routine can
-- select fewer columns than the generic rank permits; a column count is not
-- accepted here unless the returned matrix is an exact, complete kernel.
-- A supplied Lambda is still only a candidate: it is not searched for here.

load "Macaulay2/ext1_constructive.m2";

gtMPProd = (A, B) -> matrix table(numRows A, numColumns B,
    (i, j) -> sum(numColumns A, k -> A_(i,k) * B_(k,j)));

gtMPIsZero = M -> all(flatten entries M, entry -> entry == 0);

-- If the augmented operator is [L | -I_q], its complete kernel is the
-- explicit split matrix [I_p ; L].  This avoids a Gröbner syzygy search for
-- identity-completed Stafford blocks while retaining the same operator-order
-- convention as gtMPProd.
gtIdentityCompletedKernel = L -> (
    D := ring L;
    id_(D^(numColumns L)) || L
);

gtMPComplete = (R, Q) -> (
    D := ring R;
    p := numColumns R;
    Rt := gtTau R;
    IQ := image map(D^p, , entries gtTau Q);
    IS := image map(D^p, , entries syz Rt);
    isSubset(IQ, IS) and isSubset(IS, IQ)
);

gtMinimalParametrization = (R, Lambda) -> (
    if numRows Lambda != numRows R then
        error "gtMinimalParametrization expects Lambda with numRows R rows";
    P := R | (-Lambda);
    Pt := gtTau P;
    Q := gtTau mingens image syz Pt;
    expectedRank := numColumns P - numRows P;
    relationZero := gtMPIsZero gtMPProd(P, Q);
    complete := gtMPComplete(P, Q);
    if not relationZero then
        error "syzygy output does not annihilate the augmented operator";
    if not complete then
        error "syzygy output is not a complete parametrisation";
    if numColumns Q < expectedRank then
        error "parametrisation has fewer columns than the generic rank";
    new HashTable from {
        "P" => P,
        "Lambda" => Lambda,
        "Q" => Q,
        "columns" => numColumns Q,
        "genericRank" => expectedRank,
        "minimumColumnsCertified" => numColumns Q == expectedRank,
        "relationZero" => relationZero,
        "complete" => complete,
        "scope" =>
            "complete kernel over the supplied Lambda; column minimality is certified only when columns = generic rank"
    }
);

-- Bounded diagnostic for hard systems.  A SyzygyLimit is not a completeness
-- proof; this routine therefore never labels its output as a parametrisation.
gtBoundedParametrizationFromLambda = (R, Lambda, limit) -> (
    if numRows Lambda != numRows R then
        error "gtBoundedParametrizationFromLambda expects Lambda with numRows R rows";
    if limit < 1 then error "SyzygyLimit must be positive";
    P := R | (-Lambda);
    Pt := gtTau P;
    partial := gtTau mingens image syz(Pt, SyzygyLimit => limit);
    new HashTable from {
        "P" => P,
        "Lambda" => Lambda,
        "Q" => partial,
        "columns" => numColumns partial,
        "genericRank" => numColumns P - numRows P,
        "relationZero" => gtMPIsZero gtMPProd(P, partial),
        "complete" => false,
        "scope" => "bounded syzygy sample; completeness is intentionally undecided"
    }
);
