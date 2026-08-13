-- Exact negative oracle for projectivity of a presented differential module.
--
-- For a q-by-n presentation matrix P, the presented left module is
-- D^(1-by-n) / D^(1-by-q)P.  Under the full-row-rank hypothesis used in the
-- paper, projectivity is equivalent to splitting of the presentation map.
-- A split requires a right inverse S with P*S = id_q.  Consequently, the
-- exact module inclusion image(id_q) <= image(P) rules out projectivity when
-- it fails.

needsPackage "Dmodules";

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
