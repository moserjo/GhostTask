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

-- A square block C is unimodular in the operator order when Cinv is both a
-- left and a right inverse.  Keeping this check separate is important:
-- ordinary Macaulay2 matrix multiplication reverses Weyl-operator order.
gtMPUnimodularBlock = (C, Cinv) -> (
    if numRows C != numColumns C or numRows Cinv != numRows C or
       numColumns Cinv != numColumns C then false
    else (
        I := id_((ring C)^(numRows C));
        gtMPIsZero (gtMPProd(C, Cinv) - I) and
        gtMPIsZero (gtMPProd(Cinv, C) - I)
    )
);

-- If [A | C] contains an explicitly invertible q x q block C, its complete
-- kernel is obtained without a syzygy search:
--
--     [ I_p ]
--     [-Cinv*A].
--
-- This is the constructive identity-block step used before Stafford/
-- Quadrat--Robertz reduction.  The returned matrix is complete by the two
-- inverse identities, not merely a list of annihilating columns.
gtMPCompletedKernel = (A, C, Cinv) -> (
    D := ring A;
    q := numRows A;
    p := numColumns A;
    if numRows C != q or numColumns C != q then
        error "gtMPCompletedKernel expects a square block with numRows A rows";
    if ring C =!= D or ring Cinv =!= D then
        error "gtMPCompletedKernel expects matrices over the same Weyl ring";
    if not gtMPUnimodularBlock(C, Cinv) then
        error "gtMPCompletedKernel requires a checked operator-order inverse";
    id_(D^p) || (-gtMPProd(Cinv, A))
);

-- The identity-block special case [L | -I_q].
gtIdentityCompletedKernel = L -> (
    D := ring L;
    I := id_(D^(numRows L));
    gtMPCompletedKernel(L, -I, -I)
);

-- Build the data for a reduction that retains the columns in `kept` from C
-- and constrains every other C-coordinate to zero.  If Qfull is the kernel
-- of [A | C], the retained kernel is the syzygy module of the auxiliary rows
-- of Qfull, lifted through its retained rows.  This is the matrix form of
-- the unimodular completion step in the effective parametrisation algorithm.
gtMPCompletionData = (A, C, Cinv, kept) -> (
    q := numRows A;
    p := numColumns A;
    if #kept < 1 or #kept >= q then
        error "gtMPCompletionData expects a nonempty proper C-column subset";
    if any(kept, j -> j < 0 or j >= q) then
        error "gtMPCompletionData received an invalid C-column index";
    if #unique kept != #kept then
        error "gtMPCompletionData received duplicate C-column indices";
    auxiliary := select(toList(0..q-1),
        j -> not any(kept, k -> j == k));
    Qfull := gtMPCompletedKernel(A, C, Cinv);
    keptRows := toList(0..p-1) | apply(kept, j -> p + j);
    auxiliaryRows := apply(auxiliary, j -> p + j);
    P := A | submatrix(C, kept);
    new HashTable from {
        "P" => P,
        "C" => C,
        "Cinv" => Cinv,
        "kept" => kept,
        "auxiliary" => auxiliary,
        "Qfull" => Qfull,
        "keptRows" => keptRows,
        "auxiliaryRows" => auxiliaryRows,
        "constraint" => Qfull^auxiliaryRows,
        "fullRelationZero" => gtMPIsZero gtMPProd(A | C, Qfull)
    }
);

-- Compute a complete kernel after the completion reduction.  The native
-- syzygy call is made in the transposed Weyl convention and converted back
-- before the operator-order checks.  Use the bounded companion below when a
-- caller wants a diagnostic rather than an unbounded computation.
gtMPReducedKernel = (A, C, Cinv, kept) -> (
    data := gtMPCompletionData(A, C, Cinv, kept);
    S := gtTau mingens image syz (gtTau data#"constraint");
    Q := gtMPProd((data#"Qfull")^(data#"keptRows"), S);
    relationZero := gtMPIsZero gtMPProd(data#"P", Q);
    complete := gtMPComplete(data#"P", Q);
    if not relationZero then
        error "completion-reduced syzygies do not annihilate the target";
    if not complete then
        error "completion-reduced syzygies are not a complete kernel";
    new HashTable from {
        "P" => data#"P",
        "Q" => Q,
        "constraint" => data#"constraint",
        "columns" => numColumns Q,
        "genericRank" => numColumns(data#"P") - numRows(data#"P"),
        "relationZero" => relationZero,
        "complete" => complete,
        "scope" =>
            "complete kernel obtained by syzygies after unimodular completion"
    }
);

-- A bounded reduced computation is useful for hard rows, but its output is
-- never promoted to a parametrisation certificate.
gtBoundedReducedKernel = (A, C, Cinv, kept, limit) -> (
    if limit < 1 then error "SyzygyLimit must be positive";
    data := gtMPCompletionData(A, C, Cinv, kept);
    S := gtTau mingens image syz(gtTau data#"constraint",
        SyzygyLimit => limit);
    Q := gtMPProd((data#"Qfull")^(data#"keptRows"), S);
    new HashTable from {
        "P" => data#"P",
        "Q" => Q,
        "constraint" => data#"constraint",
        "columns" => numColumns Q,
        "genericRank" => numColumns(data#"P") - numRows(data#"P"),
        "relationZero" => gtMPIsZero gtMPProd(data#"P", Q),
        "complete" => false,
        "scope" =>
            "bounded completion-reduced sample; completeness is intentionally undecided"
    }
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
