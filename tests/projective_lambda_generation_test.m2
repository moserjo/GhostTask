-- Behavioral regression test for finite-space projective Lambda generation.
-- Run from the repository root with:
--   M2 --no-readline tests/projective_lambda_generation_test.m2

load "Macaulay2/projectivity_certificate.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

-- The first experiment is a bounded negative case after the nontriviality
-- guard is applied. Its small ansatz has projective two-column augmentations,
-- but they are PIGP-equivalent because the ghost block is invertible.
W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};
space1 = finiteOperatorSpace({1,dt},{-1,0,1});
g1 = generateProjectiveLambda(R1, space1, 2);
assertTrue("Experiment 1 nontrivial generator correctly fails", not g1#"found");
assertTrue("Experiment 1 failure is complete in space", g1#"completeWithinSpace");

-- A deliberately too-small finite space gives a bounded negative result,
-- not a false positive and not an implicit fallback.
small1 = generateProjectiveLambda(R1, {0,1}, 1);
assertTrue("bounded negative result is explicit", not small1#"found");
assertTrue("bounded negative result is marked complete", small1#"completeWithinSpace");

W2 = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2 = matrix{{dt^2*l+g,0,0,-1},
            {0,dt^2*l+g,0,-1},
            {0,0,dt^2*l+g,-1}};
space2 = finiteOperatorSpace({1,dt,t,l},{0,1});
g2 = generateProjectiveLambda(R2, space2, 2);
assertTrue("Experiment 2 generator succeeds", g2#"found");
assertTrue("Experiment 2 generator is complete in space", g2#"completeWithinSpace");
assertTrue("Experiment 2 generated Lambda is projective", g2#"proof"#"projective");
assertTrue("Experiment 2 generated Lambda is nontrivial",
    not columnImageFull g2#"Lambda");
assertTrue("Experiment 2 minimum ghost count is certified", g2#"minimumColumns" == 2);

W3 = QQ[x,y,z,dx,dy,dz,nu0,
    WeylAlgebra => {x=>dx,y=>dy,z=>dz}];
curl = matrix{{0,-dz,dy},{dz,0,-dx},{-dy,dx,0}};
nu = matrix{{nu0,0,0},{0,nu0,0},{0,0,z}};
R3 = curl*nu*curl | matrix{{0},{0},{-1}};
space3 = finiteOperatorSpace({1},{0,1,-1});
g3 = generateProjectiveLambda(R3, space3, 2);
assertTrue("Experiment 3 generator succeeds", g3#"found");
assertTrue("Experiment 3 generator is complete in space", g3#"completeWithinSpace");
assertTrue("Experiment 3 generated Lambda is projective", g3#"proof"#"projective");
assertTrue("Experiment 3 generated Lambda is nontrivial",
    not columnImageFull g3#"Lambda");
assertTrue("Experiment 3 minimum ghost count is certified", g3#"minimumColumns" == 2);

print("ALL FINITE-SPACE GENERATION TESTS PASSED");
