-- Macaulay2/ghosttask_systems.m2
--
-- The three paper systems as they are actually used by the running code,
-- in one machine-readable place, together with the ghost columns and the
-- hand-transcribed sympy kernels of the writing_kernel.py files.
--
-- Provenance of every matrix below:
--
--   Experiment 1  Experiment1_Pedagogical/Macaulay2_Pedagogical.txt (R and
--                 the ghost column of the RGT3 branch) and
--                 Experiment1_Pedagogical/{grid_evaluation,optimisation/GT}/
--                 writing_kernel.py (the kernel B).
--   Experiment 2  Experiment2_Tripendulum/Macaulay2_Tripendulum.txt and
--                 Experiment2_Tripendulum/grid_evaluation/writing_kernel.py.
--   Experiment 3  Experiment3_Magnetostatics/OreModules_Magnetostatics_correct.mw
--                 (curl, nu, R, R1, Exti, A_part, H) and
--                 Experiment3_Magnetostatics/{grid_evaluation,optimization/GT}/
--                 writing_kernel.py.  The stale
--                 Macaulay2_Magnetostatics_needsupdate.txt is NOT the live
--                 system; see Macaulay2/PIPELINE.md.
--
-- Naming: gtR<i> is the operator before ghost augmentation, gtLambda<i>
-- the ghost column(s) actually in use, gtB<i> the kernel that
-- writing_kernel.py hands to PCGP.  Products are operator order.

load "Macaulay2/ext1_constructive.m2";

-- Operator-order matrix product: (A B)_(l,j) = sum_i A_(l,i) * B_(i,j),
-- each factor composed in the written order.  Macaulay2's own matrix
-- product composes entries in the opposite order and must not be used
-- for these identities.
gtProd = (A, B) -> matrix table(numRows A, numColumns B,
    (l, j) -> sum(numColumns A, i -> A_(l,i) * B_(i,j)));

gtIsZero = M -> all(flatten entries M, e -> e == 0);

-- B is a COMPLETE parametrisation of R when its columns span the whole
-- syzygy module, not merely a piece of it.  This is the property the
-- "R' = transpose mingens image syz transpose B" line in the paper's
-- Macaulay2 transcripts is inspected for by eye; here it is decided.
gtIsCompleteParametrisation = (R, B) -> (
    D := ring R;
    p := numColumns R;
    Rt := gtTau R;
    IB := image map(D^p, , entries gtTau B);
    IS := image map(D^p, , entries syz Rt);
    isSubset(IB, IS) and isSubset(IS, IB)
    );

load "Macaulay2/minimal_parametrization.m2";

------------------------------------------------------------------
-- Experiment 1: pedagogical shear transport
------------------------------------------------------------------
gtW1 = QQ[t, x, dt, dx, a, WeylAlgebra => {x => dx, t => dt}];
gtR1 = matrix{{x*dx + dt, a*dx}, {0, x*dx + dt}};
gtLambda1 = matrix{{-1}, {-dt}};
gtRGT1 = gtR1 | gtLambda1;
-- writing_kernel.py, with D[0] = dx, D[1] = dt, x[0] = x
gtB1 = matrix{
    {-a*dt*dx + x*dx + dt - 1},
    {x*dt*dx + dt^2 - dt},
    {x^2*dx^2 + 2*x*dt*dx + dt^2 - dt}};
-- Certified ext^1 generator for the same R; not the column in use.
gtLambda1free = matrix{{-1}, {-t}};

------------------------------------------------------------------
-- Experiment 2: tripendulum.  l and g are fitted positive reals at run
-- time, so QQ(l,g) is the ring the numerical code lives in; QQ[l,g] is
-- kept because generation is ring dependent here.
------------------------------------------------------------------
gtW2 = QQ[t, dt, l, g, WeylAlgebra => {t => dt}];
gtR2 = matrix{
    {dt^2*l + g, 0, 0, -1},
    {0, dt^2*l + g, 0, -1},
    {0, 0, dt^2*l + g, -1}};
gtLambda2 = matrix{{l}, {l*dt}, {l*t}};
gtRGT2 = gtR2 | gtLambda2;
-- writing_kernel.py, with X[0] = dt, t[0] = t, length = l
gtB2 = matrix{
    {1_gtW2, -l^2*dt^2 - g*l},
    {1_gtW2, -dt^3*l^2 - dt*g*l},
    {1_gtW2, -t*l^2*dt^2 + 2*l^2*dt - g*l*t},
    {dt^2*l + g, 0_gtW2},
    {0_gtW2, dt^4*l^2 + 2*dt^2*g*l + g^2}};

gtK2 = frac(QQ[l, g]);
gtW2f = gtK2[t, dt, WeylAlgebra => {t => dt}];
gtR2f = sub(gtR2, gtW2f);
gtLambda2f = sub(gtLambda2, gtW2f);

------------------------------------------------------------------
-- Experiment 3: anisotropic magnetostatics.
--
-- Reluctivity, as in the live worksheet: nu = nu0 * diag(1, 1, z).
-- (methods.tex and the stale Macaulay2 transcript say diag(nu0,nu0,z);
-- the two agree only at nu0 = 1.  The kernel in writing_kernel.py is
-- reproduced by the worksheet form and not by the methods.tex form.)
--
-- L = curl nu curl is NOT full row rank: div curl = 0 is a left
-- relation.  Full row rank is restored by the J_x source column, which
-- is why the ghost-column theory applies to R1 = [L | -e1] and not to L.
--
-- Tasks of the GT kernel, in order: A1, A2, A3, J_x, gauge, B1, B2, B3.
-- Rows 1-5 are the parametrisation of the 3 x 5 augmented operator;
-- rows 6-8 are curl of rows 1-3 and carry no extra latent process.
------------------------------------------------------------------
gtK3 = frac(QQ[nu0]);
gtW3 = gtK3[x, y, z, dx, dy, dz, WeylAlgebra => {x => dx, y => dy, z => dz}];
gtNu0 = sub(nu0, gtK3);
gtCurl = matrix{{0, -dz, dy}, {dz, 0, -dx}, {-dy, dx, 0}};
gtNu = matrix{{gtNu0*1_gtW3, 0, 0}, {0, gtNu0*1_gtW3, 0}, {0, 0, gtNu0*z}};
gtL3 = gtProd(gtCurl, gtProd(gtNu, gtCurl));
gtSource3 = matrix{{-1_gtW3}, {0}, {0}};          -- J_x task, equation 1
gtR3 = gtL3 | gtSource3;
gtLambda3 = matrix{{0_gtW3}, {-1}, {0}};          -- gauge ghost task, equation 2
gtRGT3 = gtR3 | gtLambda3;
-- writing_kernel.py Magnetostatics_GT, D[0] = dx, D[1] = dy, D[2] = dz,
-- x[2] = z.  Eight rows; the first five are the parametrisation.
gtB3full = matrix{
    {-dx, -dy},
    {-dy, dx},
    {-dz, 0_gtW3},
    {0_gtW3, gtNu0*dy*(dz^2 + z*(dx^2 + dy^2))},
    {0_gtW3, -gtNu0*dx*(dz^2 + z*(dx^2 + dy^2))},
    {0_gtW3, -dz*dx},
    {0_gtW3, -dz*dy},
    {0_gtW3, dx^2 + dy^2}};
gtB3 = gtB3full^{0,1,2,3,4};
gtA3part = gtB3full^{0,1,2};

-- Certified one-column alternative for Experiment 3, found by the
-- oracle: this ghost column makes the augmentation stably free, which
-- the gauge column -e2 does not.  See Macaulay2/PIPELINE.md.
gtLambda3free = matrix{{0_gtW3}, {x*z}, {1}};
