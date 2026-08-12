-- Constructive, bounded search for Ghost Tasking augmentations.
--
-- This file deliberately separates three claims which were previously mixed:
--
--   1. transpose(B)*transpose(P) = 0: B is a complete relation matrix;
--   2. P is recovered from B by a second syzygy calculation;
--   3. the column image of P is full (a diagnostic for the identity
--      baseline, not the stable-free certificate in the Ghost Tasking proof).
--
-- The search is complete only inside the finite operator ansatz supplied by
-- the caller.  A failed search is therefore a failure of that ansatz, not a
-- counterexample to the two-generator theorem.

needsPackage "Dmodules";

columnMatrix = entries -> matrix apply(entries, entry -> {entry});

allOperatorVectors = (q, operators) -> (
    if q == 0 then {{}}
    else (
        tails := allOperatorVectors(q - 1, operators);
        flatten apply(operators, operator ->
            apply(tails, tail -> prepend(operator, tail)))
    )
);

isZeroColumn = entries -> (# select(entries, entry -> entry != 0) == 0);

doubleAnnihilatorCertificate = P -> (
    B := Dtransposition mingens image syz Dtransposition P;
    recoveredP := transpose mingens image syz transpose B;
    relationZero := transpose B * transpose P == 0;
    exact := isSubset(image transpose P, image transpose recoveredP)
        and isSubset(image transpose recoveredP, image transpose P);
    new HashTable from {
        "B" => B,
        "recoveredP" => recoveredP,
        "relationZero" => relationZero,
        "exact" => exact
    }
);

-- This is only a column-image diagnostic in M2's left-module representation.
-- It is intentionally reported separately from `exact`: it is true for the
-- identity baseline, but it is not the row-module/stable-free condition used
-- in the Ghost Tasking proof.
columnImageFull = P -> (
    isSubset(image id_(target P), image P)
);

checkLambda = (R, Lambda) -> (
    P := R | (-Lambda);
    certificate := doubleAnnihilatorCertificate P;
    new HashTable from {
        "P" => P,
        "Lambda" => Lambda,
        "B" => certificate#"B",
        "recoveredP" => certificate#"recoveredP",
        "relationZero" => certificate#"relationZero",
        "exact" => certificate#"exact",
        "columnImageFull" => columnImageFull P
    }
);

-- Search one-column candidates first, then pairs of columns.  The operator
-- list is finite and ordered, so the result is reproducible.  The returned
-- `tested` count makes the discovery effort auditable.
searchLambda = (R, operators, maxColumns) -> (
    if maxColumns < 0 or maxColumns > 2 then
        error "searchLambda supports maxColumns = 0, 1, or 2";
    q := numRows R;
    candidates := allOperatorVectors(q, operators);
    tested := 0;
    found := null;

    if maxColumns >= 1 then (
        for i from 0 to #candidates - 1 do (
            if found === null and not isZeroColumn(candidates#i) then (
                tested = tested + 1;
                result := checkLambda(R, columnMatrix(candidates#i));
                if result#"exact" then found = result;
            )
        )
    );

    if found === null and maxColumns >= 2 then (
        for i from 0 to #candidates - 1 do (
            for j from i + 1 to #candidates - 1 do (
                if found === null and
                    not isZeroColumn(candidates#i) and
                    not isZeroColumn(candidates#j) then (
                    tested = tested + 1;
                    result := checkLambda(R,
                        columnMatrix(candidates#i) | columnMatrix(candidates#j));
                    if result#"exact" then found = result;
                )
            )
        )
    );

    if found === null then
        new HashTable from {
            "found" => false,
            "tested" => tested,
            "candidateCount" => #candidates
        }
    else (
        new HashTable from {
            "found" => true,
            "tested" => tested,
            "candidateCount" => #candidates,
            "P" => found#"P",
            "Lambda" => found#"Lambda",
            "B" => found#"B",
            "recoveredP" => found#"recoveredP",
            "relationZero" => found#"relationZero",
            "exact" => found#"exact",
            "columnImageFull" => found#"columnImageFull"
        }
    )
);
