-- Regression tests for Macaulay2/constructive_lambda.m2.
-- Run from the repository root with:
--   M2 --no-readline tests/constructive_lambda_test.m2

load "Macaulay2/constructive_lambda.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

checkFound = (label, certificate, expectedColumns) -> (
    assertTrue(label | " found", certificate#"found");
    assertTrue(label | " direct relation check",
        (transpose certificate#"B") * (transpose certificate#"P") == 0);
    assertTrue(label | " relation flag", certificate#"relationZero");
    assertTrue(label | " exact double-annihilator check", certificate#"exact");
    assertTrue(label | " column count",
        numColumns certificate#"Lambda" == expectedColumns);
    assertTrue(label | " column-image diagnostic is distinct",
        not certificate#"columnImageFull")
);

-- Experiment 1: two independent first-order operators.
W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};
baseline1 = checkLambda(R1, id_(target R1));
assertTrue("Experiment 1 identity baseline is exact", baseline1#"exact");
assertTrue("Experiment 1 identity column image is full", baseline1#"columnImageFull");
assertTrue("Experiment 1 original system is not exact", not (doubleAnnihilatorCertificate R1)#"exact");
c1 = searchLambda(R1, {0,1,dt}, 1);
checkFound("Experiment 1", c1, 1);
assertTrue("Experiment 1 bounded count", c1#"candidateCount" == 9);
assertTrue("Experiment 1 finds the first nonzero candidate", c1#"tested" == 1);

-- Experiment 2: tripendulum operator.  The ansatz includes the operators
-- appearing in the known one-column augmentation, but discovery is ordered
-- and exhaustive over all 7^3 columns in this finite set.
W2 = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2 = matrix{{dt^2*l+g,0,0,-1},
            {0,dt^2*l+g,0,-1},
            {0,0,dt^2*l+g,-1}};
baseline2 = checkLambda(R2, id_(target R2));
assertTrue("Experiment 2 identity baseline is exact", baseline2#"exact");
assertTrue("Experiment 2 identity column image is full", baseline2#"columnImageFull");
c2 = searchLambda(R2, {0,1,dt,t,l,l*dt,l*t}, 1);
checkFound("Experiment 2", c2, 1);
assertTrue("Experiment 2 bounded count", c2#"candidateCount" == 343);
assertTrue("Experiment 2 tests only ten candidates before success", c2#"tested" == 10);

-- Experiment 3: magnetostatics operator with the gauge/source column.
W3 = QQ[x,y,z,dx,dy,dz,nu0,
    WeylAlgebra => {x=>dx,y=>dy,z=>dz}];
curl = matrix{{0,-dz,dy},{dz,0,-dx},{-dy,dx,0}};
nu = matrix{{nu0,0,0},{0,nu0,0},{0,0,z}};
R3 = curl*nu*curl | matrix{{0},{0},{-1}};
baseline3 = checkLambda(R3, id_(target R3));
assertTrue("Experiment 3 identity baseline is exact", baseline3#"exact");
assertTrue("Experiment 3 identity column image is full", baseline3#"columnImageFull");
c3 = searchLambda(R3, {0,1,-1}, 1);
checkFound("Experiment 3", c3, 1);
assertTrue("Experiment 3 bounded count", c3#"candidateCount" == 27);
assertTrue("Experiment 3 tests three candidates before success", c3#"tested" == 3);

print("ALL CONSTRUCTIVE LAMBDA TESTS PASSED");
