-- Behavioral regression test for the positive projectivity proof oracle.
-- Run from the repository root with:
--   M2 --no-readline tests/projectivity_proof_test.m2

load "Macaulay2/projectivity_certificate.m2";

assertTrue = (label, condition) -> (
    if not condition then error("FAIL: " | label);
    print("PASS: " | label)
);

-- The positive witnesses use only constant ghost columns.  The operator
-- blocks are arbitrary: the proof is the checked right-inverse identity.
W1 = QQ[t,x,dt,dx,a, WeylAlgebra => {t=>dt,x=>dx}];
R1 = matrix{{x*dx+dt,a*dx},{0,x*dx+dt}};
L1 = matrix{{0,1},{1,0}};
P1 = R1 | (-L1);
o1 = projectivityProofOracle P1;
assertTrue("Experiment 1 proof oracle succeeds", o1#"projective");
assertTrue("Experiment 1 witness is checked", P1*o1#"witness" == id_(target P1));

W2 = QQ[t,dt,l,g, WeylAlgebra => {t=>dt}];
R2 = matrix{{dt^2*l+g,0,0,-1},
            {0,dt^2*l+g,0,-1},
            {0,0,dt^2*l+g,-1}};
L2 = matrix{{0,0},{0,1},{1,0}};
P2 = R2 | (-L2);
o2 = projectivityProofOracle P2;
assertTrue("Experiment 2 proof oracle succeeds", o2#"projective");
assertTrue("Experiment 2 witness is checked", P2*o2#"witness" == id_(target P2));

W3 = QQ[x,y,z,dx,dy,dz,nu0,
    WeylAlgebra => {x=>dx,y=>dy,z=>dz}];
curl = matrix{{0,-dz,dy},{dz,0,-dx},{-dy,dx,0}};
nu = matrix{{nu0,0,0},{0,nu0,0},{0,0,z}};
R3 = curl*nu*curl | matrix{{0},{0},{-1}};
L3 = matrix{{0,1},{1,0},{0,0}};
P3 = R3 | (-L3);
o3 = projectivityProofOracle P3;
assertTrue("Experiment 3 proof oracle succeeds", o3#"projective");
assertTrue("Experiment 3 witness is checked", P3*o3#"witness" == id_(target P3));

-- A deliberately corrupted witness must not be accepted.
bad = projectivityProofWithWitness(P1, matrix{{0,0},{0,0},{0,0},{0,0}});
assertTrue("corrupted witness rejected", not bad#"projective");

print("ALL POSITIVE PROJECTIVITY TESTS PASSED");
