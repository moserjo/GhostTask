-- Macaulay2/stafford_reduce.m2
--
-- OUR OWN Stafford reduction: from the q standard generators of the
-- torsion right module
--
--     ext^1_D(M, D) = D^q / (R D^p),      D = A_n(QQ),
--
-- down to TWO generators, as guaranteed by Stafford, JLMS (2) 18 (1978),
-- introduction item (3) p. 429 (Theorem 3.7 p. 438 composed with Theorem
-- 3.1 p. 434).
--
-- WHY A SECOND IMPLEMENTATION.  Not because the shipped Maple
-- `StaffordReduction` of Quadrat and Robertz is broken.  It is not.
-- Recorded stalls elsewhere were a call site omitting
-- `"reduce_generators"=true`; see Macaulay2/PIPELINE.md.  This file
-- exists because
-- two independent constructors answering to one oracle is a stronger
-- position than either alone, because it removes the Maple dependency
-- from the construction path, and because it does not enter the
-- `OreRank` torsion audit at all -- torsion follows from full row rank
-- and the verifier decides it independently -- which turns several
-- recorded cost boundaries into answers.
--
-- WHAT IS AND IS NOT CLAIMED.  This file is a SEARCH.  It is not a proof
-- of anything, and a failure of the search is a bounded statement about
-- the declared candidate pool and the declared degree bound, never a
-- counterexample to Stafford's theorem.  Every POSITIVE outcome, by
-- contrast, is decided by the independent Groebner oracle
-- `lambdaGeneratesExt1Decision` of
-- Macaulay2/ext1_constructive.m2 and is therefore a
-- theorem for the displayed R and the displayed Lambda.  Each
-- intermediate block of the reduction is certified by the same oracle,
-- so the reduction chain is certified step by step and not only at the
-- end.
--
-- THE ALGORITHM.  Stafford's Theorem 3.1 says: for generators m_1, m_2,
-- m_3 of a right ideal and arbitrary nonzero c_1, c_2 there are f, g
-- with (m_1 + m_3 f c_1, m_2 + m_3 g c_2) generating the same ideal.  The
-- proof is an existence proof; the effective versions (Quadrat-Robertz,
-- Acta Appl. Math. 133 (2014), Theorem 9) search a bounded pool for f
-- and g.  We do the same but with a pool we control and enlarge:
--
--   stage 0  can a column simply be dropped?
--   stage 1  Stafford form: replace columns (a, b, c) by
--            (G_a + G_c f, G_b + G_c g), f, g normal-ordered monomials
--            of total degree <= deg, over triples (a,b,c) with a < b
--   stage 2  random q x (r-1) recombination with a FIXED seed, entries
--            drawn from the same pool
--
-- with deg increased 0, 1, 2, ... up to a declared bound.  Termination
-- with success is not guaranteed by us; Stafford's theorem guarantees
-- that some f, g exist, only the degree bound is unknown.  Exhausting the
-- bound is reported as EXHAUSTED, a cost boundary.
--
-- CONVENTIONS, and why they matter here.  ext^1 is a RIGHT module: a
-- D-combination of the columns G_1..G_r is sum_i G_i c_i with the scalar
-- on the RIGHT, so entry (l,j) of a recombined block is
-- sum_i G_(l,i) * U_(i,j) -- the G entry LEFT of the U entry, an honest
-- operator product in that written order.  Macaulay2's own matrix
-- product composes entries in the OPPOSITE order, so `G*U` is the wrong
-- thing here and is never used; `srProd` below builds the product
-- entrywise in the operator order.  A convention slip could only make a
-- generating block look non-generating to the oracle, never the reverse.

load "Macaulay2/ext1_constructive.m2";

-- Operator-order matrix product: (A B)_(l,j) = sum_i A_(l,i) * B_(i,j),
-- each product taken in the written order in the Weyl algebra.
srProd = (A, B) -> matrix table(numRows A, numColumns B,
    (l, j) -> sum(numColumns A, i -> A_(l,i) * B_(i,j)));

-- All normal-ordered monomials x^a d^b of total degree exactly `deg` in
-- the Weyl algebra D, together with the constant 1 at deg = 0.  Normal
-- ordering is automatic: in QQ[x.., d.., WeylAlgebra => ..] the x
-- variables precede the d variables, so a monomial in the internal
-- basis already has every polynomial coefficient to the left.
-- (Macaulay2's `basis` refuses to run over a Weyl algebra, so the
-- exponent vectors are enumerated directly.)
srCompositions = (len, deg) -> (
    if len == 0 then return if deg == 0 then {{}} else {};
    flatten apply(deg + 1, e ->
        apply(srCompositions(len - 1, deg - e), t -> prepend(e, t)))
    );

srMonomialsOfDegree = (D, deg) -> (
    g := gens D;
    apply(srCompositions(#g, deg), e ->
        product(#g, i -> (g#i)^(e#i)))
    );

-- Cumulative pool up to total degree deg.
srPool = (D, deg) -> flatten apply(deg + 1, k -> srMonomialsOfDegree(D, k));

-- Does the block G (columns in D^q) generate D^q/(R D^p)?
srGenerates = (R, G) -> lambdaGeneratesExt1Decision(R, G);

-- Column c of G, right-multiplied by f, added into column a.
srCombine = (G, a, c, f) -> matrix table(numRows G, 1,
    (l, j) -> G_(l,a) + G_(l,c) * f);

-- One reduction step: r columns -> r-1 columns, certified.  Returns the
-- new block, or null if the declared pool is exhausted.
srReduceStep = (R, G, maxdeg, verbose) -> (
    D := ring R;
    r := numColumns G;
    if r <= 2 then return G;
    idx := toList(0 .. r-1);
    -- stage 0: is one of the columns redundant?
    for a in idx do (
        Gp := G_(select(idx, i -> i != a));
        if srGenerates(R, Gp) then (
            if verbose then
                << "    step " << r << " -> " << r-1
                   << ": column " << a+1 << " was redundant" << endl;
            return Gp;
            );
        );
    -- stage 1: Stafford form over triples and a growing pool.  Only
    -- a < b is enumerated: swapping a and b permutes the two new
    -- columns of the SAME block, and generation does not see column
    -- order, so the ordered enumeration would test every candidate
    -- exactly twice.
    for deg from 0 to maxdeg do (
        pool := srPool(D, deg);
        newOnly := srMonomialsOfDegree(D, deg);
        for a in idx do for b in idx do for c in idx do (
            if a >= b or a == c or b == c then continue;
            rest := select(idx, i -> i != a and i != b and i != c);
            for f in pool do for g in pool do (
                -- only look at pairs that are new at this degree
                if deg > 0 and not (member(f, newOnly) or member(g, newOnly))
                    then continue;
                Gp := G_rest | srCombine(G, a, c, f) | srCombine(G, b, c, g);
                if srGenerates(R, Gp) then (
                    if verbose then
                        << "    step " << r << " -> " << r-1
                           << ": (a,b,c) = (" << a+1 << "," << b+1 << ","
                           << c+1 << "), f = " << f << ", g = " << g
                           << "  [deg <= " << deg << "]" << endl;
                    return Gp;
                    );
                );
            );
        );
    -- stage 2: random recombination, fixed seed, growing pool
    setRandomSeed 20260821;
    for deg from 0 to maxdeg do (
        pool := srPool(D, deg);
        for trial from 1 to 200 do (
            U := matrix table(r, r-1, (i,j) -> pool#(random(#pool)));
            Gp := srProd(G, U);
            if srGenerates(R, Gp) then (
                if verbose then
                    << "    step " << r << " -> " << r-1
                       << ": random recombination, trial " << trial
                       << " [deg <= " << deg << "]" << endl;
                return Gp;
                );
            );
        );
    null
    );

-- Full reduction q -> 2.  Returns a HashTable with "status" one of
--   "ext1zero"   ext^1 = 0, nothing to generate
--   "trivial"    q <= 2, the standard basis already is the block
--   "reduced"    a certified q x 2 block was constructed
--   "exhausted"  the declared pool and degree bound did not suffice
srReduceToTwo = method(Options => {MaxDegree => 3, Verbose => true, Torsion => false});
srReduceToTwo Matrix := opts -> R -> (
    D := ring R;
    q := numRows R;
    -- Hypothesis audit, cheapest first.  Full row rank is one syzygy
    -- computation.  Torsion of ext^1 FOLLOWS from full row rank -- that
    -- is the theorem this construction serves, machine-checked in
    -- proofs/two_column_general.lean -- so recomputing it here would
    -- cost a characteristic-variety dimension for no new information.
    -- The independent verifier decides torsion anyway, from the spec and
    -- not from us, so the audit is not weakened by moving it there.  Set
    -- Torsion => true to run it here as well; that is what the negative
    -- control does, since a matrix without full row rank is exactly the
    -- case where the implication is unavailable.
    if not gtRowInjective R then (
        if opts.Torsion and ext1IsTorsion R then
            return new HashTable from {"status" => "torsionwithoutrank"};
        return new HashTable from {"status" => "notfullrowrank"};
        );
    if opts.Torsion and not ext1IsTorsion R then
        return new HashTable from {"status" => "nottorsion"};
    if ext1IsZero R then
        return new HashTable from {"status" => "ext1zero"};
    G := id_(D^q);
    if q <= 2 then (
        Lam := if q == 2 then G else G | map(D^1, D^1, 0);
        return new HashTable from {
            "status" => "trivial", "Lambda" => Lam,
            "certified" => srGenerates(R, Lam), "steps" => 0};
        );
    steps := 0;
    while numColumns G > 2 do (
        Gn := srReduceStep(R, G, opts.MaxDegree, opts.Verbose);
        if Gn === null then
            return new HashTable from {
                "status" => "exhausted", "Lambda" => G, "steps" => steps};
        G = Gn;
        steps = steps + 1;
        );
    new HashTable from {
        "status" => "reduced", "Lambda" => G,
        "certified" => srGenerates(R, G), "steps" => steps}
    );
