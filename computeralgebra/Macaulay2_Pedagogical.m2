
needsPackage "Dmodules"
--Alg = QQ[t, x, dt, dx, a, WeylAlgebra => {x=>dx, t=> dt}]
kk = frac(QQ[a]);
Alg = kk[t, x, dt, dx, WeylAlgebra => {x=>dx, t=> dt}]

--defining helper functions
combos = (S,n) -> if n == 0 then {{}} else flatten apply(S, s -> apply(combos(S,n-1), c -> prepend(s,c)));
strip = X -> map((ring X)^(numRows X), (ring X)^(numColumns X), entries X)

-- defining checks
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

-- defining brute force search
findTorsionFreeLambda = (R,ops) -> (D := ring R; space := apply(ops, o -> promote(o,D)); 
                        L := null; 
                        scan(combos(space,numRows R), 
                        v -> if L === null and any(v, e -> e != 0) 
                        then (C := matrix apply(v, e -> {e}); 
                        if isTorsionFree(R|(-C)) then L = C));
                         L);

findProjectiveLambda = (R,ops) -> (D := ring R; space := apply(ops, o -> promote(o,D)); 
                        L := null; 
                        scan(combos(space,numRows R), 
                        v -> if L === null and any(v, e -> e != 0) 
                        then (C := matrix apply(v, e -> {e}); 
                        if isProjective(R|(-C)) then L = C));
                         L);




R = matrix{{x*dx + dt, a*dx}, {0, x*dx + dt}} --system matrix
isTorsionFree(R) --check if torsion free: no

ext1M = coker Dtransposition R;
isHolonomic ext1M --is not holonomic, so we cannot directly calculate Lambda, instead we brute force (or guess)

Lambda = findTorsionFreeLambda(R, {0,1,-1,dt,-dt}) --non unique
LambdaUsed = matrix{{1},{dt}} --Lambda used in paper

P = R|(-LambdaUsed) --define new system

isTorsionFree(P) --check if torsion free
isProjective(P) --check if projective 

--calculating parametrization matrix Q:
Q = Dtransposition mingens image syz Dtransposition P --not necessarily minimal
transpose Q*(transpose P) -- test: should be 0
--alternative torsion-free check
Pnew = transpose mingens image syz transpose Q --if the rows have the same linear span as the rows of P, it's torsion free



--to find a projective Lambda, we can also apply brute force search 
LambdaProjective = findProjectiveLambda(R, {0,1,-1, t, -t})

PProj = R|(-LambdaProjective)
isTorsionFree(PProj) --check if torsion free: yes
isProjective(PProj) --check if projective : yes
RightInverse(PProj) --alternative check: if right inverse exists, it is also projective

--calculating parametrization matrix Q:
QProj = Dtransposition mingens image syz Dtransposition PProj --not necessarily minimal
transpose QProj*(transpose PProj) -- test: should be 0
--alternative torsion-free check
PProjnew = transpose mingens image syz transpose Q --if the rows have the same linear span as the rows of P, it's torsion free
