-- Behavioral regression test for the negative projectivity oracle.
-- Run from the repository root with:
--   M2 --no-readline tests/projectivity_certificate_test.m2

load "Macaulay2/projectivity_certificate.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

-- Experiment 1: the published one-column candidate is parametrizing, but
-- its presented module does not split and is therefore not projective.
W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};
bad1 = negativeProjectivityCertificate (R1 | (-matrix{{0},{1}}));
assertTrue("Experiment 1 sparse Lambda has full row rank", bad1#"rowInjective");
assertTrue("Experiment 1 sparse Lambda is ruled out", bad1#"ruledOut");
assertTrue("Experiment 1 no right inverse", not bad1#"rightInverseExists");
assertTrue("Experiment 1 residual is nonzero", bad1#"residual" != 0);

-- Experiment 2.
W2 = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2 = matrix{{dt^2*l+g,0,0,-1},
            {0,dt^2*l+g,0,-1},
            {0,0,dt^2*l+g,-1}};
bad2 = negativeProjectivityCertificate (R2 | (-matrix{{0},{1},{t}}));
assertTrue("Experiment 2 sparse Lambda has full row rank", bad2#"rowInjective");
assertTrue("Experiment 2 sparse Lambda is ruled out", bad2#"ruledOut");
assertTrue("Experiment 2 no right inverse", not bad2#"rightInverseExists");

-- Experiment 3.
W3 = QQ[x,y,z,dx,dy,dz,nu0,
    WeylAlgebra => {x=>dx,y=>dy,z=>dz}];
curl = matrix{{0,-dz,dy},{dz,0,-dx},{-dy,dx,0}};
nu = matrix{{nu0,0,0},{0,nu0,0},{0,0,z}};
R3 = curl*nu*curl | matrix{{0},{0},{-1}};
bad3 = negativeProjectivityCertificate (R3 | (-matrix{{0},{1},{0}}));
assertTrue("Experiment 3 sparse Lambda has full row rank", bad3#"rowInjective");
assertTrue("Experiment 3 sparse Lambda is ruled out", bad3#"ruledOut");
assertTrue("Experiment 3 no right inverse", not bad3#"rightInverseExists");

print("ALL NEGATIVE PROJECTIVITY TESTS PASSED");
