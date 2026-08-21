-- tests/ghosttask_pipeline_test.m2
--
-- Behavioural test of the paper pipeline itself, not of a helper.
--
-- Part A re-derives, over the Weyl algebra, the three kernels that
-- writing_kernel.py hands to PCGP, and asserts for each that it is an
-- exact operator-order syzygy of the augmented operator AND a complete
-- parametrisation of it.  That is the step the repository currently
-- performs by copying Maple or Macaulay2 output into sympy by hand.
--
-- Part B constructs certified ghost blocks with Macaulay2/stafford_reduce.m2
-- and certifies each one with the unmodified oracle
-- lambdaGeneratesExt1Decision of Macaulay2/ext1_constructive.m2.
--
-- Part C records which of the ghost columns in use generate ext^1, i.e.
-- for which experiments the augmented presentation is stably free.
-- Torsion-freeness (Part A) is what the GP needs; stable freeness is
-- strictly stronger and is not implied by it.
--
-- Run from the repository root:
--   M2 --script tests/ghosttask_pipeline_test.m2

load "Macaulay2/ghosttask_systems.m2";
load "Macaulay2/stafford_reduce.m2";

failures = 0;
assertTrue = (name, ok) -> (
    if ok then << "PASS: " << name << endl
    else (<< "FAIL: " << name << endl; failures = failures + 1));

------------------------------------------------------------------
-- Part A: the transcribed kernels
------------------------------------------------------------------
assertTrue("Ex1 kernel annihilates [R | Lambda] in operator order",
    gtIsZero gtProd(gtRGT1, gtB1));
assertTrue("Ex1 kernel is a complete parametrisation",
    gtIsCompleteParametrisation(gtRGT1, gtB1));

assertTrue("Ex2 kernel annihilates [R | Lambda] in operator order",
    gtIsZero gtProd(gtRGT2, gtB2));
assertTrue("Ex2 kernel is a complete parametrisation",
    gtIsCompleteParametrisation(gtRGT2, gtB2));

assertTrue("Ex3 kernel rows 1-5 annihilate [L | -e1 | -e2]",
    gtIsZero gtProd(gtRGT3, gtB3));
assertTrue("Ex3 kernel rows 1-5 are a complete parametrisation",
    gtIsCompleteParametrisation(gtRGT3, gtB3));
-- The last three tasks are the magnetic field, curl of the first three.
assertTrue("Ex3 kernel rows 6-8 are curl of rows 1-3",
    entries gtProd(gtCurl, gtA3part) == entries gtB3full^{5,6,7});
-- Rows 4-5 are the first two components of L A; the third vanishes
-- identically, which is why the kernel has eight rows and not nine.
assertTrue("Ex3 kernel rows 4-5 are (L A)_1, (L A)_2 with (L A)_3 = 0",
    entries gtProd(gtL3, gtA3part) ==
        entries (gtB3full^{3,4} || matrix{{0_gtW3, 0}}));

-- The reluctivity really is nu0*diag(1,1,z) and not diag(nu0,nu0,z):
-- with the second form the same identity fails.
gtNuAlt = matrix{{gtNu0*1_gtW3, 0, 0}, {0, gtNu0*1_gtW3, 0}, {0, 0, z}};
gtLalt = gtProd(gtCurl, gtProd(gtNuAlt, gtCurl));
assertTrue("Ex3 kernel is NOT reproduced by nu = diag(nu0,nu0,z)",
    entries gtProd(gtLalt, gtA3part) =!=
        entries (gtB3full^{3,4} || matrix{{0_gtW3, 0}}));

-- L alone has a left relation (div curl = 0); the source column is what
-- makes the full-row-rank hypothesis of Stafford 3.7 available.
assertTrue("Ex3 L = curl nu curl is not full row rank",
    not gtRowInjective gtL3);
assertTrue("Ex3 [L | -e1] is full row rank", gtRowInjective gtR3);
assertTrue("Ex1 R is full row rank", gtRowInjective gtR1);
assertTrue("Ex2 R is full row rank", gtRowInjective gtR2);

------------------------------------------------------------------
-- Part B: constructed and certified ghost blocks
------------------------------------------------------------------
reduceAndCertify = (name, R) -> (
    t0 := cpuTime();
    out := srReduceToTwo(R, MaxDegree => 2, Verbose => false);
    dt := cpuTime() - t0;
    << "  " << name << ": status " << out#"status"
       << ", cpu " << dt << " s";
    if out#?"Lambda" then (
        << ", Lambda = " << toString entries out#"Lambda";
        << endl;
        assertTrue(name | " constructed block certified by the oracle",
            lambdaGeneratesExt1Decision(R, out#"Lambda"));
        ) else << endl;
    out);

<< "-- constructed ghost blocks --" << endl;
reduceAndCertify("Ex1", gtR1);
reduceAndCertify("Ex2 over QQ[l,g]", gtR2);
reduceAndCertify("Ex2 over QQ(l,g)", gtR2f);
reduceAndCertify("Ex3 over QQ(nu0)", gtR3);

------------------------------------------------------------------
-- Part C: do the ghost columns in use generate ext^1?
------------------------------------------------------------------
-- Experiment 2 over the ring the numerics live in: yes.
assertTrue("Ex2 ghost column (l, l dt, l t) generates ext^1 over QQ(l,g)",
    lambdaGeneratesExt1Decision(gtR2f, gtLambda2f));
-- Over QQ[l,g] it does not: the witnesses need 1/l.
assertTrue("Ex2 the same column does not generate over QQ[l,g]",
    not lambdaGeneratesExt1Decision(gtR2, gtLambda2));
-- Experiment 1: the column in use does not generate; (-1,-t) does.
assertTrue("Ex1 ghost column (-1,-dt) does not generate ext^1",
    not lambdaGeneratesExt1Decision(gtR1, gtLambda1));
assertTrue("Ex1 (-1,-t) does generate ext^1",
    lambdaGeneratesExt1Decision(gtR1, gtLambda1free));
-- Experiment 3: the gauge column does not generate; (0, xz, 1) does.
assertTrue("Ex3 gauge ghost column -e2 does not generate ext^1",
    not lambdaGeneratesExt1Decision(gtR3, gtLambda3));
assertTrue("Ex3 (0, xz, 1) generates ext^1 over QQ(nu0)",
    lambdaGeneratesExt1Decision(gtR3, gtLambda3free));
assertTrue("Ex3 (0, xz, 1) makes the augmentation stably free",
    ext1IsZero (gtR3 | (-gtLambda3free)));
-- Near misses, so the certificate is seen to discriminate.
assertTrue("Ex3 (0, 1, 0) rejected",
    not lambdaGeneratesExt1Decision(gtR3, matrix{{0_gtW3},{1},{0}}));
assertTrue("Ex3 (0, 0, 1) rejected",
    not lambdaGeneratesExt1Decision(gtR3, matrix{{0_gtW3},{0},{1}}));
assertTrue("Ex3 (1, xz, 0) rejected on the live system",
    not lambdaGeneratesExt1Decision(gtR3, matrix{{1_gtW3},{x*z},{0}}));

if failures == 0 then print("ALL GHOSTTASK PIPELINE TESTS PASSED") else (
    print(toString failures | " FAILURES");
    exit 1);
