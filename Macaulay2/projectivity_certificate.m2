-- Exact negative oracle for projectivity of a presented differential module.
--
-- For a q-by-n presentation matrix P, the presented left module is
-- D^(1-by-n) / D^(1-by-q)P.  Under the full-row-rank hypothesis used in the
-- paper, projectivity is equivalent to splitting of the presentation map.
-- A split requires a right inverse S with P*S = id_q.  Consequently, the
-- exact module inclusion image(id_q) <= image(P) rules out projectivity when
-- it fails.

needsPackage "Dmodules";

gtColumnMatrix = entries -> matrix apply(entries, entry -> {entry});

gtAllOperatorVectors = (q, operators) -> (
    if q == 0 then { { } }
    else (
        tails := gtAllOperatorVectors(q - 1, operators);
        flatten apply(operators, operator ->
            apply(tails, tail -> prepend(operator, tail)))
    )
);

gtIsZeroColumn = entries -> (# select(entries, entry -> entry != 0) == 0);

rowInjective = P -> (
    -- D-transposition converts left row relations into the right syzygy
    -- convention used by Macaulay2.
    numColumns (syz transpose Dtransposition P) == 0
);

columnImageFull = P -> (
    isSubset(image id_(target P), image P)
);

ruleOutProjectivity = P -> (
    rowInjective P and not columnImageFull P
);

negativeProjectivityCertificate = P -> (
    injective := rowInjective P;
    full := columnImageFull P;
    -- `I // P` is the exact factor returned by Macaulay2's module
    -- Gröbner-basis computation.  When the image is not full, the residual
    -- records the failed identity factorization found by that computation.
    I := id_(target P);
    partialFactor := I // P;
    residual := I - P*partialFactor;
    new HashTable from {
        "P" => P,
        "rowInjective" => injective,
        "rightInverseExists" => full,
        "ruledOut" => injective and not full,
        "partialFactor" => partialFactor,
        "residual" => residual,
        "criterion" => "full-row-rank presentation splits iff P has a right inverse"
    }
);

-- Positive oracle.  A caller may provide an explicit S, or ask Macaulay2
-- to compute the exact factor I // P.  The returned witness is accepted only
-- after the identity P*S == id_q is checked in the original Weyl algebra.
projectivityProofOracle = P -> (
    injective := rowInjective P;
    I := id_(target P);
    witness := I // P;
    split := witness =!= null and P*witness == I;
    new HashTable from {
        "P" => P,
        "rowInjective" => injective,
        "witness" => witness,
        "split" => split,
        "projective" => injective and split,
        "proof" => if injective and split then
            "row-injective presentation plus checked right inverse"
            else "no projectivity proof returned"
    }
);

projectivityProofWithWitness = (P, S) -> (
    injective := rowInjective P;
    split := P*S == id_(target P);
    new HashTable from {
        "P" => P,
        "S" => S,
        "rowInjective" => injective,
        "split" => split,
        "projective" => injective and split,
        "proof" => if injective and split then
            "explicit right-inverse identity checked over the Weyl algebra"
            else "supplied witness rejected"
    }
);

-- Turn a finite basis and finite coefficient alphabet into an explicit finite
-- operator space.  This is intentionally finite: an operator space over a
-- field has infinitely many coefficients, so exhaustive generation requires
-- the coefficient alphabet to be part of the certificate.
finiteOperatorSpace = (basis, coefficients) -> (
    values := flatten apply(basis, b -> apply(coefficients, c -> c*b));
    unique values
);

-- Exhaustive constructive generation inside a declared finite operator
-- space.  The result is guaranteed only relative to that finite space and
-- requested column bound.  It returns no Lambda rather than silently using
-- an identity augmentation.
generateProjectiveLambda = (R, operatorSpace, maxColumns) -> (
    if maxColumns < 1 or maxColumns > 2 then
        error "generateProjectiveLambda supports maxColumns = 1 or 2";
    q := numRows R;
    candidates := gtAllOperatorVectors(q, operatorSpace);
    tested := 0;
    found := null;

    if maxColumns >= 1 then (
        for i from 0 to #candidates - 1 do (
            if found === null and not gtIsZeroColumn(candidates#i) then (
                tested = tested + 1;
                Lambda := gtColumnMatrix(candidates#i);
                P := R | (-Lambda);
                proof := projectivityProofOracle P;
                if proof#"projective" then found = new HashTable from {
                    "Lambda" => Lambda,
                    "P" => P,
                    "proof" => proof
                };
            );
        );
    );

    if found === null and maxColumns == 2 then (
        for i from 0 to #candidates - 1 do for j from i + 1 to #candidates - 1 do (
            if found === null and
                not gtIsZeroColumn(candidates#i) and
                not gtIsZeroColumn(candidates#j) then (
                tested = tested + 1;
                Lambda := gtColumnMatrix(candidates#i) | gtColumnMatrix(candidates#j);
                P := R | (-Lambda);
                proof := projectivityProofOracle P;
                if proof#"projective" then found = new HashTable from {
                    "Lambda" => Lambda,
                    "P" => P,
                    "proof" => proof
                };
            );
        );
    );

    if found === null then new HashTable from {
        "found" => false,
        "tested" => tested,
        "spaceSize" => #operatorSpace,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true
    } else new HashTable from {
        "found" => true,
        "tested" => tested,
        "spaceSize" => #operatorSpace,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true,
        "Lambda" => found#"Lambda",
        "P" => found#"P",
        "proof" => found#"proof"
    }
);
