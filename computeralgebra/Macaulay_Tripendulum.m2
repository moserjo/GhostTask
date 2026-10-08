
needsPackage "Dmodules"
kk = frac(QQ[l,g]);
W = kk[t,dt, WeylAlgebra => {t=>dt}];

--defining functions
strip = X -> map((ring X)^(numRows X), (ring X)^(numColumns X), entries X)

isTorsionFree = P -> (
    K1 := strip syz P;
    Q1 := strip transpose Dtransposition P;
    Q2 := strip transpose Dtransposition K1;
    ext1N := prune (kernel Q2 / image Q1);
    ext1N == 0)
isProjective = P -> isSubset(image id_(target Dtransposition P), image Dtransposition P);
RightInverse = P -> (
W = id_(target Dtransposition P) // Dtransposition P;
W =!= null and (Dtransposition P)*W == id_(target Dtransposition P);
S = Dtransposition W) --this is the right inverse


R = matrix{{l*dt^2+g,0,0,-1},{0,l*dt^2+g,0,-1},{0,0,l*dt^2+g,-1}};
isTorsionFree(R) --check if torsion free: no

ext1M = coker Dtransposition R;
isHolonomic ext1M --if true, the next line works
Lambda = Dtransposition (makeCyclic Dtransposition R).Generator
--gives out non-unique solution, (guessed) alternative used subsequently is (1, dt, t)

LambdaUsed = matrix{{-1}, {-dt}, {-t}}
P = R|(-LambdaUsed) --define new system





isTorsionFree(P) --check if torsion free: yes
isProjective(P) --check if projective (here: equivalent)
-- alternative check: calculating if a right inverse exists
RightInverse(P)

--calculating parametrization matrix Q:
Q = Dtransposition mingens image syz Dtransposition P
transpose Q*(transpose P) -- test: should be 0

--alternative torsion-free check
Pnew = transpose mingens image syz transpose Q --if the rows have the same linear span as the rows of P, it's torsion free

