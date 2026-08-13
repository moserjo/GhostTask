-- Behavioral regression test for the constructive extension-module layer.
-- Run from the repository root with:
--   M2 --no-readline tests/ext1_constructive_test.m2
--
-- All identities are asserted in the operator order
-- (P*S)_ij = sum_k P_ik S_kj by re-multiplying in the tau-world, which
-- is independent of the code path that produced the witness.  Family
-- memberships are checked by reassembling Lambda from the returned
-- (U, X), and negative cases assert certificate failure.

load "Macaulay2/projectivity_certificate.m2";
load "Macaulay2/ext1_constructive.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

-- independent operator-order product check
trueRightInverseHolds = (P, S) -> (
    Pt := Dtransposition P;
    St := Dtransposition S;
    Pt*St == id_(target Pt)
);

----------------------------------------------------------------------
-- Experiment 1: shear-transport system over the Weyl algebra QQ[a](x,t).
----------------------------------------------------------------------
W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};

assertTrue("Ex1 ext^1 is nonzero", not ext1IsZero R1);
assertTrue("Ex1 ext^1 is torsion", ext1IsTorsion R1);
assertTrue("Ex1 ext^1 is not holonomic", not ext1IsHolonomic R1);

L1 = matrix{{-1},{-t}};
cert1 = lambdaGeneratesExt1(R1, L1);
assertTrue("Ex1 (-1,-t) generates ext^1", cert1#"generates");
assertTrue("Ex1 P is row injective", cert1#"rowInjective");
assertTrue("Ex1 operator-order witness identity",
    trueRightInverseHolds(R1 | (-L1), cert1#"witness"));

min1 = minimumColumnReport(R1, L1);
assertTrue("Ex1 minimum r = 1 is exact", min1#"minimumIsExact");

-- The full one-column family: every admissible column decomposes as
-- L1*u + R1*X with u generating D/K1.  The eight literal ansatz
-- solutions from the finite search must all be members.
literal1 = {
    matrix{{1},{t}}, matrix{{1},{-t}}, matrix{{-1},{t}}, matrix{{-1},{-t}},
    matrix{{t},{1}}, matrix{{t},{-1}}, matrix{{-t},{1}}, matrix{{-t},{-1}}};
scan(literal1, L -> (
    m := lambdaFamilyMembership(R1, L1, L);
    assertTrue("Ex1 member " | toString flatten entries L,
        m#"decomposed" and m#"lambdaGenerates" and m#"consistent");
));

-- Non-generators must fail the oracle.  The paper's original Ex1 guess
-- (1, dt) renders the system parametrizable but does not split it, so
-- it is correctly rejected here.
scan({matrix{{0},{1}}, matrix{{1},{dt}}}, L -> (
    c := lambdaGeneratesExt1(R1, L);
    assertTrue("Ex1 non-generator rejected " | toString flatten entries L,
        not c#"generates");
));

-- Normal forms mod R separate residue classes.
nf1a = lambdaNormalForm(R1, matrix{{1},{t}});
nf1b = lambdaNormalForm(R1, matrix{{t},{1}});
assertTrue("Ex1 normal forms separate (1,t) from (t,1)",
    entries nf1a != entries nf1b);

----------------------------------------------------------------------
-- Experiment 2: tripendulum.  Ring dependence of the minimum.
----------------------------------------------------------------------
-- Over the first Weyl algebra with coefficient field QQ(l,g) the
-- extension module is holonomic, hence cyclic: one ghost column
-- suffices.  Both the simple column (-1,-t,0) and the paper's original
-- guess (-1,-dt,-t) are certified generators.
kk = frac(QQ[l,g]);
W2 = kk[t,dt, WeylAlgebra => {t=>dt}];
R2 = matrix{{l*dt^2+g,0,0,-1},{0,l*dt^2+g,0,-1},{0,0,l*dt^2+g,-1}};

assertTrue("Ex2 ext^1 nonzero over QQ(l,g)", not ext1IsZero R2);
assertTrue("Ex2 ext^1 holonomic over QQ(l,g)", ext1IsHolonomic R2);

L2 = matrix{{-1},{-t},{0_W2}};
cert2 = lambdaGeneratesExt1(R2, L2);
assertTrue("Ex2 (-1,-t,0) generates ext^1 over QQ(l,g)", cert2#"generates");
assertTrue("Ex2 P is row injective", cert2#"rowInjective");
assertTrue("Ex2 operator-order witness identity",
    trueRightInverseHolds(R2 | (-L2), cert2#"witness"));

L2paper = matrix{{-1},{-dt},{-t}};
cert2p = lambdaGeneratesExt1(R2, L2paper);
assertTrue("Ex2 paper guess (-1,-dt,-t) generates over QQ(l,g)",
    cert2p#"generates");

min2 = minimumColumnReport(R2, L2);
assertTrue("Ex2 minimum r = 1 is exact over QQ(l,g)", min2#"minimumIsExact");

m2paper = lambdaFamilyMembership(R2, L2, L2paper);
assertTrue("Ex2 paper guess is a member of the (-1,-t,0) family",
    m2paper#"decomposed" and m2paper#"lambdaGenerates" and m2paper#"consistent");

-- Over the polynomial parameter ring QQ[l,g] the same column fails: the
-- witnesses need 1/g.  The two-column block stays certified there, so
-- the minimum is ring-dependent.
W2poly = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2poly = matrix{{l*dt^2+g,0,0,-1},{0,l*dt^2+g,0,-1},{0,0,l*dt^2+g,-1}};
cert2fail = lambdaGeneratesExt1(R2poly, matrix{{-1},{-t},{0_W2poly}});
assertTrue("Ex2 (-1,-t,0) does not generate over QQ[l,g]",
    not cert2fail#"generates");
L2two = matrix{{0_W2poly,0},{0,1},{1,0}};
cert2two = lambdaGeneratesExt1(R2poly, L2two);
assertTrue("Ex2 two-column block generates over QQ[l,g]",
    cert2two#"generates");
assertTrue("Ex2 two-column operator-order witness identity",
    trueRightInverseHolds(R2poly | (-L2two), cert2two#"witness"));
assertTrue("Ex2 two-column block is not PIGP-trivial",
    not lambdaIsPIGPTrivial L2two);

print("ALL EXT1 CONSTRUCTIVE TESTS PASSED");
