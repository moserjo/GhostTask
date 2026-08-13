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

-- Any Lambda whose columns already generate the target is PIGP-equivalent:
-- the ghost block itself has a right inverse.  This excludes the identity
-- and all invertible or redundant variants of that trivial augmentation.
gtIsPIGPTrivial = Lambda -> columnImageFull Lambda;

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

gtOperatorSpaceInRing = (R, operatorSpace) -> (
    D := ring first flatten entries R;
    apply(operatorSpace, operator -> promote(operator,D))
);

-- Exhaustive constructive generation inside a declared finite operator
-- space.  The result is guaranteed only relative to that finite space and
-- requested column bound.  It returns no Lambda rather than silently using
-- an identity augmentation.
generateProjectiveLambda = (R, operatorSpace, maxColumns) -> (
    if maxColumns < 1 or maxColumns > 2 then
        error "generateProjectiveLambda supports maxColumns = 1 or 2";
    q := numRows R;
    operatorSpaceInRing := gtOperatorSpaceInRing(R, operatorSpace);
    candidates := gtAllOperatorVectors(q, operatorSpaceInRing);
    tested := 0;
    found := null;
    foundColumns := null;

    if maxColumns >= 1 then (
        for i from 0 to #candidates - 1 do (
            if found === null and not gtIsZeroColumn(candidates#i) then (
                tested = tested + 1;
                Lambda := gtColumnMatrix(candidates#i);
                if not gtIsPIGPTrivial Lambda then (
                    P := R | (-Lambda);
                    proof := projectivityProofOracle P;
                    if proof#"projective" then (
                        found = new HashTable from {
                            "Lambda" => Lambda,
                            "P" => P,
                            "proof" => proof
                        };
                        foundColumns = 1;
                    );
                );
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
                if not gtIsPIGPTrivial Lambda then (
                    P := R | (-Lambda);
                    proof := projectivityProofOracle P;
                    if proof#"projective" then (
                        found = new HashTable from {
                            "Lambda" => Lambda,
                            "P" => P,
                            "proof" => proof
                        };
                        foundColumns = 2;
                    );
                );
            );
        );
    );

    if found === null then new HashTable from {
        "found" => false,
        "tested" => tested,
        "spaceSize" => #operatorSpaceInRing,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true,
        "minimumColumns" => null,
        "ghostColumns" => 0,
        "solutionCount" => 0,
        "uniqueWithinSpace" => false,
        "nontrivialityGuard" => "reject Lambda with full column image"
    } else new HashTable from {
        "found" => true,
        "tested" => tested,
        "spaceSize" => #operatorSpaceInRing,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true,
        "Lambda" => found#"Lambda",
        "P" => found#"P",
        "proof" => found#"proof",
        "minimumColumns" => foundColumns,
        "ghostColumns" => foundColumns,
        "solutionCount" => null,
        "uniqueWithinSpace" => null,
        "nontrivialityGuard" => "reject Lambda with full column image"
    }
);

-- Exhaustively enumerate every nontrivial projective Lambda at the minimum
-- column count in a finite operator space.  This can be much more expensive
-- than generateProjectiveLambda, which stops at the first witness.
enumerateMinimalProjectiveLambdas = (R, operatorSpace, maxColumns) -> (
    if maxColumns < 1 or maxColumns > 2 then
        error "enumerateMinimalProjectiveLambdas supports maxColumns = 1 or 2";
    q := numRows R;
    operatorSpaceInRing := gtOperatorSpaceInRing(R, operatorSpace);
    candidates := gtAllOperatorVectors(q, operatorSpaceInRing);
    tested := 0;
    solutions := {};
    minimumColumns := null;

    for i from 0 to #candidates - 1 do (
        if not gtIsZeroColumn(candidates#i) then (
            tested = tested + 1;
            Lambda := gtColumnMatrix(candidates#i);
            if not gtIsPIGPTrivial Lambda then (
                P := R | (-Lambda);
                proof := projectivityProofOracle P;
                if proof#"projective" then solutions = append(solutions,
                    new HashTable from {
                        "Lambda" => Lambda,
                        "P" => P,
                        "proof" => proof
                    });
            );
        );
    );
    if #solutions > 0 then minimumColumns = 1;

    if #solutions == 0 and maxColumns == 2 then (
        for i from 0 to #candidates - 1 do for j from i + 1 to #candidates - 1 do (
            if not gtIsZeroColumn(candidates#i) and
                not gtIsZeroColumn(candidates#j) then (
                tested = tested + 1;
                Lambda := gtColumnMatrix(candidates#i) | gtColumnMatrix(candidates#j);
                if not gtIsPIGPTrivial Lambda then (
                    P := R | (-Lambda);
                    proof := projectivityProofOracle P;
                    if proof#"projective" then solutions = append(solutions,
                        new HashTable from {
                            "Lambda" => Lambda,
                            "P" => P,
                            "proof" => proof
                        });
                );
            );
        );
        if #solutions > 0 then minimumColumns = 2;
    );

    if #solutions == 0 then new HashTable from {
        "found" => false,
        "tested" => tested,
        "spaceSize" => #operatorSpaceInRing,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true,
        "minimumColumns" => null,
        "solutions" => {},
        "solutionCount" => 0,
        "uniqueWithinSpace" => false,
        "nontrivialityGuard" => "reject Lambda with full column image"
    } else new HashTable from {
        "found" => true,
        "tested" => tested,
        "spaceSize" => #operatorSpaceInRing,
        "candidateCount" => #candidates,
        "maxColumns" => maxColumns,
        "completeWithinSpace" => true,
        "minimumColumns" => minimumColumns,
        "solutions" => apply(solutions, solution -> solution#"Lambda"),
        "certificates" => solutions,
        "solutionCount" => #solutions,
        "uniqueWithinSpace" => #solutions == 1,
        "nontrivialityGuard" => "reject Lambda with full column image"
    }
);

-- Enumerate a finite Weyl-algebra ball.  The ball contains all words in the
-- supplied operator generators up to maxWordLength, multiplied by the
-- supplied finite coefficient alphabet.  This is a constructive search over
-- an explicitly declared finite subset, not an identity fallback.
operatorBall = (operatorGenerators, coefficientAlphabet, maxWordLength) -> (
    if #operatorGenerators == 0 then error "operatorBall needs generators";
    if maxWordLength < 0 then error "maxWordLength must be nonnegative";
    R := ring first operatorGenerators;
    oneOperator := promote(1,R);
    words := {oneOperator};
    frontier := {oneOperator};
    for degree from 1 to maxWordLength do (
        frontier = flatten apply(frontier, word ->
            apply(operatorGenerators, generator -> word*generator));
        words = join(words, frontier);
    );
    unique(flatten apply(words, word ->
        apply(coefficientAlphabet, coefficient -> promote(coefficient,R)*word)))
);

-- Search the finite operator ball and retain the finite completeness claim
-- from generateProjectiveLambda.  Increasing the word length and coefficient
-- alphabet gives a semi-decision procedure for a successful Lambda, but a
-- bounded failure is not a global no-go theorem.
generalProjectiveLambdaSearch =
    (R, operatorGenerators, coefficientAlphabet, maxWordLength, maxColumns) -> (
        operatorSpace := operatorBall(
            operatorGenerators, coefficientAlphabet, maxWordLength);
        bounded := generateProjectiveLambda(R, operatorSpace, maxColumns);
        if bounded#"found" then new HashTable from {
            "found" => true,
            "Lambda" => bounded#"Lambda",
            "P" => bounded#"P",
            "proof" => bounded#"proof",
            "tested" => bounded#"tested",
            "spaceSize" => bounded#"spaceSize",
            "candidateCount" => bounded#"candidateCount",
            "maxColumns" => maxColumns,
            "maxWordLength" => maxWordLength,
            "operatorSpace" => operatorSpace,
            "completeWithinOperatorBall" => true,
            "globalStatus" =>
                "semi-decision by enlarging operator and coefficient balls",
            "identityExcluded" => true,
            "pigpTrivialExcluded" => true,
            "minimumColumns" => bounded#"minimumColumns",
            "ghostColumns" => bounded#"ghostColumns",
            "minimumCertifiedWithinOperatorBall" => true
        } else new HashTable from {
            "found" => false,
            "tested" => bounded#"tested",
            "spaceSize" => bounded#"spaceSize",
            "candidateCount" => bounded#"candidateCount",
            "maxColumns" => maxColumns,
            "maxWordLength" => maxWordLength,
            "operatorSpace" => operatorSpace,
            "completeWithinOperatorBall" => true,
            "globalStatus" =>
                "semi-decision by enlarging operator and coefficient balls",
            "identityExcluded" => true,
            "pigpTrivialExcluded" => true,
            "minimumColumns" => null,
            "ghostColumns" => 0,
            "minimumCertifiedWithinOperatorBall" => true
        }
    );

-- General-ball version of enumerateMinimalProjectiveLambdas.  It returns the
-- complete set of literal minimal solutions in the requested finite ball.
generalProjectiveLambdaSolutions =
    (R, operatorGenerators, coefficientAlphabet, maxWordLength, maxColumns) -> (
        operatorSpace := operatorBall(
            operatorGenerators, coefficientAlphabet, maxWordLength);
        bounded := enumerateMinimalProjectiveLambdas(R, operatorSpace, maxColumns);
        if bounded#"found" then new HashTable from {
            "found" => true,
            "solutions" => bounded#"solutions",
            "certificates" => bounded#"certificates",
            "tested" => bounded#"tested",
            "spaceSize" => bounded#"spaceSize",
            "candidateCount" => bounded#"candidateCount",
            "maxColumns" => maxColumns,
            "maxWordLength" => maxWordLength,
            "operatorSpace" => operatorSpace,
            "minimumColumns" => bounded#"minimumColumns",
            "solutionCount" => bounded#"solutionCount",
            "uniqueWithinOperatorBall" => bounded#"uniqueWithinSpace",
            "completeWithinOperatorBall" => true,
            "globalStatus" =>
                "semi-decision by enlarging operator and coefficient balls",
            "identityExcluded" => true,
            "pigpTrivialExcluded" => true
        } else new HashTable from {
            "found" => false,
            "solutions" => {},
            "tested" => bounded#"tested",
            "spaceSize" => bounded#"spaceSize",
            "candidateCount" => bounded#"candidateCount",
            "maxColumns" => maxColumns,
            "maxWordLength" => maxWordLength,
            "operatorSpace" => operatorSpace,
            "minimumColumns" => null,
            "solutionCount" => 0,
            "uniqueWithinOperatorBall" => false,
            "completeWithinOperatorBall" => true,
            "globalStatus" =>
                "semi-decision by enlarging operator and coefficient balls",
            "identityExcluded" => true,
            "pigpTrivialExcluded" => true
        }
    );
