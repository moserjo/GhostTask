-- Behavioral regression test for generated Weyl-algebra operator balls.
-- Run from the repository root with:
--   M2 --no-readline tests/general_lambda_generation_test.m2

load "Macaulay2/projectivity_certificate.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};
unit1 = promote(1,W1);
g1 = generalProjectiveLambdaSearch(R1, {unit1}, {-1,0,1}, 0, 2);
assertTrue("general ball excludes Experiment 1 PIGP case", not g1#"found");
assertTrue("Experiment 1 ball exclusion is complete", g1#"completeWithinOperatorBall");
assertTrue("identity augmentation is excluded", g1#"identityExcluded");
assertTrue("PIGP-equivalent augmentation is excluded", g1#"pigpTrivialExcluded");

W2 = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2 = matrix{{dt^2*l+g,0,0,-1},
            {0,dt^2*l+g,0,-1},
            {0,0,dt^2*l+g,-1}};
g2 = generalProjectiveLambdaSearch(R2, {t,dt,l}, {0,1}, 1, 2);
assertTrue("general ball finds Experiment 2 Lambda", g2#"found");
assertTrue("Experiment 2 ball result is complete", g2#"completeWithinOperatorBall");
assertTrue("Experiment 2 ball result is proven projective", g2#"proof"#"projective");
assertTrue("Experiment 2 minimum ghost count is certified", g2#"minimumColumns" == 2);

all1 = generalProjectiveLambdaSolutions(R1, {unit1}, {-1,0,1}, 0, 2);
assertTrue("all minimal Experiment 1 nontrivial solutions are absent",
    not all1#"found");
assertTrue("full finite negative result is complete",
    all1#"completeWithinOperatorBall");

-- Synthetic control: an already split presentation lets the enumerator expose
-- all three nonzero one-column choices from {0,1}. None is an identity-sized
-- or full-image Lambda.
W0 = QQ[t0,dt0, WeylAlgebra => {t0=>dt0}];
one0 = promote(1,W0);
R0 = matrix{{one0,0},{0,one0}};
all0 = generalProjectiveLambdaSolutions(R0, {dt0}, {0,1}, 0, 2);
assertTrue("full finite positive solution set is found", all0#"found");
assertTrue("minimum one-column case is certified", all0#"minimumColumns" == 1);
assertTrue("all minimal one-column solutions are reported",
    all0#"solutionCount" == 3);
assertTrue("positive finite solution set is correctly non-unique",
    not all0#"uniqueWithinOperatorBall");

small = generalProjectiveLambdaSearch(R2, {dt}, {0,1}, 0, 1);
assertTrue("general bounded failure is explicit", not small#"found");
assertTrue("general bounded failure is complete in its ball",
    small#"completeWithinOperatorBall");

print("ALL GENERAL-BALL GENERATION TESTS PASSED");
