-- Behavioural regression tests for the supplied-Lambda parametrisation API.

load "Macaulay2/ghosttask_systems.m2";

failures = 0;
assertTrue = (name, ok) -> (
    if ok then << "PASS: " << name << endl
    else (<< "FAIL: " << name << endl; failures = failures + 1));

-- These are exact complete-kernel computations, not just relation checks.
ex1 := gtMinimalParametrization(gtR1, gtLambda1free);
assertTrue("Ex1 supplied Lambda gives an exact complete kernel",
    ex1#"relationZero" and ex1#"complete");
assertTrue("Ex1 wrapper does not claim rank-one minimality",
    ex1#"columns" >= ex1#"genericRank" and
    not ex1#"minimumColumnsCertified");

ex2 := gtMinimalParametrization(gtR2f, gtLambda2f);
assertTrue("Ex2 supplied Lambda gives an exact complete kernel",
    ex2#"relationZero" and ex2#"complete");
assertTrue("Ex2 reaches its generic rank",
    ex2#"columns" == ex2#"genericRank" and ex2#"minimumColumnsCertified");

ex3 := gtMinimalParametrization(gtRGT3, gtLambda3);
assertTrue("Ex3 in-use Lambda gives an exact complete kernel",
    ex3#"relationZero" and ex3#"complete");
assertTrue("Ex3 in-use kernel reaches its generic rank",
    ex3#"columns" == ex3#"genericRank" and ex3#"minimumColumnsCertified");

-- The projective Ex3 candidate is certified as a generator, but its native
-- syzygy is the known cost boundary.  A bounded call must expose that fact,
-- never turn a partial sample into a false complete parametrisation.
assertTrue("Ex3 projective Lambda is certified independently",
    lambdaGeneratesExt1Decision(gtR3, gtLambda3free));
partial := gtBoundedParametrizationFromLambda(gtR3, gtLambda3free, 2);
assertTrue("Ex3 projective bounded result is marked incomplete",
    partial#"relationZero" and not partial#"complete");

-- The checked implementation uses Stafford's certified two-column fallback
-- for this hard projective case.  The source column and (e2,e3) complete an
-- identity block, so the kernel is constructed directly as [I3 ; L].
ex3twoP := gtR3 | (-gtLambda3two);
assertTrue("Ex3 two-column Stafford block annihilates the augmented operator",
    gtMPIsZero gtMPProd(ex3twoP, gtB3two));
assertTrue("Ex3 two-column Stafford block is a complete parametrisation",
    gtMPComplete(ex3twoP, gtB3two));
ex3two := gtMinimalParametrization(gtR3, gtLambda3two);
assertTrue("Ex3 two-column parametrisation reaches its generic rank",
    ex3two#"relationZero" and ex3two#"complete" and
    ex3two#"columns" == ex3two#"genericRank" and
    ex3two#"minimumColumnsCertified");

if failures == 0 then print("ALL MINIMAL PARAMETRIZATION TESTS PASSED") else (
    print(toString failures | " FAILURES");
    exit 1);
