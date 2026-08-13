-- Behavioral regression test for the constructive extension-module layer.
-- Run from the repository root with:
--   M2 --no-readline tests/ext1_constructive_test.m2
--
-- The oracle facts asserted here are independent of the implementation:
-- right-inverse identities are checked by explicit multiplication, family
-- memberships are checked by reassembling Lambda from the returned (U, X),
-- and negative cases assert that the corresponding certificates fail.

load "Macaulay2/projectivity_certificate.m2";
load "Macaulay2/ext1_constructive.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
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
assertTrue("Ex1 witness identity holds",
    (R1 | (-L1)) * cert1#"witness" == id_(target R1));

min1 = minimumColumnReport(R1, L1);
assertTrue("Ex1 minimum r = 1 is exact", min1#"minimumIsExact");

-- The full one-column family: every admissible column decomposes as
-- L1*u + R1*X with u generating D/K1.  The eight literal ansatz solutions
-- from the finite search must all be members with consistent certificates.
literal1 = {
    matrix{{1},{t}}, matrix{{1},{-t}}, matrix{{-1},{t}}, matrix{{-1},{-t}},
    matrix{{t},{1}}, matrix{{t},{-1}}, matrix{{-t},{1}}, matrix{{-t},{-1}}};
scan(literal1, L -> (
    m := lambdaFamilyMembership(R1, L1, L);
    assertTrue("Ex1 member " | toString flatten entries L,
        m#"decomposed" and m#"lambdaGenerates" and m#"consistent");
));

-- A non-generating column must decompose but fail the generation test.
bad1 = matrix{{0},{1}};
mbad1 = lambdaFamilyMembership(R1, L1, bad1);
assertTrue("Ex1 (0,1) decomposes but does not generate",
    mbad1#"decomposed" and not mbad1#"lambdaGenerates" and mbad1#"consistent");

-- Normal forms mod R distinguish inequivalent residue classes.
nf1a = lambdaNormalForm(R1, matrix{{1},{t}});
nf1b = lambdaNormalForm(R1, matrix{{t},{1}});
assertTrue("Ex1 normal forms separate (1,t) from (t,1)", nf1a != nf1b);

----------------------------------------------------------------------
-- Experiment 2: tripendulum.  Ring dependence of the minimum.
----------------------------------------------------------------------
-- Over the first Weyl algebra with inverted parameters (coefficient field
-- QQ(l,g)) the extension module is holonomic, hence cyclic: one ghost
-- column suffices, and both the simple column (-1,-t,0) and the paper's
-- original guess (-1,-dt,-t) are certified generators.
kk = frac(QQ[l,g]);
W2 = kk[t,dt, WeylAlgebra => {t=>dt}];
R2 = matrix{{l*dt^2+g,0,0,-1},{0,l*dt^2+g,0,-1},{0,0,l*dt^2+g,-1}};

assertTrue("Ex2 ext^1 nonzero over QQ(l,g)", not ext1IsZero R2);

L2 = matrix{{-1},{-t},{0_W2}};
cert2 = lambdaGeneratesExt1(R2, L2);
assertTrue("Ex2 (-1,-t,0) generates ext^1 over QQ(l,g)", cert2#"generates");
assertTrue("Ex2 witness identity holds",
    (R2 | (-L2)) * cert2#"witness" == id_(target R2));

L2paper = matrix{{-1},{-dt},{-t}};
cert2p = lambdaGeneratesExt1(R2, L2paper);
assertTrue("Ex2 paper guess (-1,-dt,-t) generates over QQ(l,g)",
    cert2p#"generates");

min2 = minimumColumnReport(R2, L2);
assertTrue("Ex2 minimum r = 1 is exact over QQ(l,g)", min2#"minimumIsExact");

m2paper = lambdaFamilyMembership(R2, L2, L2paper);
assertTrue("Ex2 paper guess is a member of the (-1,-t,0) family",
    m2paper#"decomposed" and m2paper#"lambdaGenerates" and m2paper#"consistent");

-- Over the polynomial parameter ring QQ[l,g] the same columns fail: the
-- witnesses need 1/g.  The two-column block from the earlier search stays
-- certified there.  The minimum is therefore ring-dependent.
W2poly = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2poly = matrix{{l*dt^2+g,0,0,-1},{0,l*dt^2+g,0,-1},{0,0,l*dt^2+g,-1}};
L2polyFail = matrix{{-1},{-t},{0_W2poly}};
cert2fail = lambdaGeneratesExt1(R2poly, L2polyFail);
assertTrue("Ex2 (-1,-t,0) does not generate over QQ[l,g]",
    not cert2fail#"generates");
L2two = matrix{{0,0},{0,1},{1,0}};
cert2two = lambdaGeneratesExt1(R2poly, L2two);
assertTrue("Ex2 two-column block generates over QQ[l,g]",
    cert2two#"generates");

print("ALL EXT1 CONSTRUCTIVE TESTS PASSED");
