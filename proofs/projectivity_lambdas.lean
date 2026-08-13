import Mathlib

/-
  Machine-checked split certificates for the ghost-tasking augmentations.

  Convention.  Lean's matrix product is the operator-order convention used in
  the paper: `(P * S) i j = ∑ k, P i k * S k j`, i.e. the entries of `P` are
  applied on the left of the entries of `S`.  Everything below lives in an
  arbitrary (noncommutative) ring `A`; no commutativity is ever assumed except
  where it is stated as a hypothesis.

  `ex1_split`, `ex2_split`, `ex3_split` verify the two-column certificates.
  Their witnesses have constant (integer) entries: the right inverse factors
  through the source columns of the augmentation, so the operator entries of
  the first block never have to be touched.  This is why the entries there may
  be left completely abstract.

  `tripendulum_split` is the genuinely operator-valued statement.  It verifies
  the tripendulum ONE-COLUMN ghost certificate over `Q(l, g)`, presented as an
  identity in any ring `A` containing the Weyl algebra `k(l,g)⟨t, ∂⟩`:  `t` and
  `d = ∂` obey `d * t = t * d + 1`, the parameters `l`, `g` are central, and
  `hg` is a central element with `hg * (2 * g) = 1`, i.e. `hg = 1/(2 g)`.  With
  `L = l * d ^ 2 + g` the pendulum operator, the certificate says
  `P2 * S2 = 1`, so the one-column augmentation is a split epimorphism.

  The proof factors the witness as `S2 = T2 * hg` with `T2` free of `hg`
  (clearing the denominator `2 g`), checks the polynomial identity
  `P2 * T2 = (2 g) * 1` by Weyl normalisation, and then divides by `2 g` once.
-/

open Matrix

namespace GhostTasking

variable {A : Type*} [Ring A]

def ex1P (a b c : A) : Matrix (Fin 2) (Fin 4) A :=
  !![a, b, 0, -1;
     0, c, -1, 0]

def ex1S : Matrix (Fin 4) (Fin 2) ℤ :=
  !![0, 0;
     0, 0;
     0, -1;
     -1, 0]

def ex2P (L : A) : Matrix (Fin 3) (Fin 6) A :=
  !![L, 0, 0, -1, 0, 0;
     0, L, 0, -1, 0, -1;
     0, 0, L, -1, -1, 0]

def ex2S : Matrix (Fin 6) (Fin 3) ℤ :=
  !![0, 0, 0;
     0, 0, 0;
     0, 0, 0;
     -1, 0, 0;
     1, 0, -1;
     1, -1, 0]

def ex3P (a b c d e f g h i : A) : Matrix (Fin 3) (Fin 6) A :=
  !![a, b, c, 0, 0, -1;
     d, e, f, 0, -1, 0;
     g, h, i, -1, 0, 0]

def ex3S : Matrix (Fin 6) (Fin 3) ℤ :=
  !![0, 0, 0;
     0, 0, 0;
     0, 0, 0;
     0, 0, -1;
     0, -1, 0;
     -1, 0, 0]

theorem ex1_split (a b c : A) :
    ex1P a b c * (ex1S.map (Int.castRingHom A)) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ex1P, ex1S, Matrix.mul_apply,
    Fin.sum_univ_succ]

theorem ex2_split (L : A) :
    ex2P L * (ex2S.map (Int.castRingHom A)) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ex2P, ex2S, Matrix.mul_apply,
    Fin.sum_univ_succ]

theorem ex3_split (a b c d e f g h i : A) :
    ex3P a b c d e f g h i * (ex3S.map (Int.castRingHom A)) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ex3P, ex3S, Matrix.mul_apply,
    Fin.sum_univ_succ]

/-! ### The tripendulum one-column certificate -/

/-- The one-column augmentation of the tripendulum system.  The diagonal
entries are the pendulum operator `L = l * d ^ 2 + g`. -/
def P2 (t d l g : A) : Matrix (Fin 3) (Fin 5) A :=
  !![l * d ^ 2 + g, 0, 0, -1, 1;
     0, l * d ^ 2 + g, 0, -1, t;
     0, 0, l * d ^ 2 + g, -1, 0]

/-- The ghost certificate: a right inverse of `P2` over `Q(l, g)`, written with
`hg = 1/(2 g)`. -/
def S2 (t d l g hg : A) : Matrix (Fin 5) (Fin 3) A :=
  !![hg * (t * d) + hg,
       -(hg * d),
       -(hg * (t * d)) + hg * d - hg;
     hg * (t ^ 2 * d) - hg * t,
       -(hg * (t * d)) + 2 * hg,
       -(hg * (t ^ 2 * d)) + hg * (t * d) + hg * t - 2 * hg;
     0, 0, 0;
     0, 0, -1;
     -(l * hg) * (t * d ^ 3) - (g * hg) * (t * d) - 3 * (l * hg) * d ^ 2 + g * hg,
       (l * hg) * d ^ 3 + (g * hg) * d,
       (l * hg) * (t * d ^ 3) - (l * hg) * d ^ 3 + (g * hg) * (t * d)
         + 3 * (l * hg) * d ^ 2 - (g * hg) * d - g * hg]

/-- The certificate with the denominator `2 g` cleared: `S2 = T2 * hg`. -/
def T2 (t d l g : A) : Matrix (Fin 5) (Fin 3) A :=
  !![t * d + 1, -d, -(t * d) + d - 1;
     t ^ 2 * d - t, -(t * d) + 2, -(t ^ 2 * d) + t * d + t - 2;
     0, 0, 0;
     0, 0, -(2 * g);
     -(l * (t * d ^ 3)) - g * (t * d) - 3 * (l * d ^ 2) + g,
       l * d ^ 3 + g * d,
       l * (t * d ^ 3) - l * d ^ 3 + g * (t * d) + 3 * (l * d ^ 2) - g * d - g]

set_option maxHeartbeats 4000000 in
/-- The tripendulum one-column ghost certificate.  `t`, `d` generate a Weyl
algebra, `l`, `g`, `hg` are central, and `hg = 1/(2 g)`. -/
theorem tripendulum_split (t d l g hg : A)
    (h_comm : d * t = t * d + 1)
    (hl_t : l * t = t * l) (hl_d : l * d = d * l)
    (hgc_t : g * t = t * g) (hgc_d : g * d = d * g)
    (hl_g : l * g = g * l)
    (hhg_t : hg * t = t * hg) (hhg_d : hg * d = d * hg)
    (hhg_l : hg * l = l * hg) (hhg_g : hg * g = g * hg)
    (h_inv : hg * (2 * g) = 1) (h_inv' : (2 * g) * hg = 1) :
    P2 t d l g * S2 t d l g hg = 1 := by
  -- numerals and powers, so that normalisation only ever sees atoms
  have n2 : ∀ x : A, 2 * x = x + x := fun x => by noncomm_ring
  have n2' : ∀ x : A, x * 2 = x + x := fun x => by noncomm_ring
  have n3 : ∀ x : A, 3 * x = x + x + x := fun x => by noncomm_ring
  have n3' : ∀ x : A, x * 3 = x + x + x := fun x => by noncomm_ring
  have p2 : ∀ x : A, x ^ 2 = x * x := fun x => by noncomm_ring
  have p3 : ∀ x : A, x ^ 3 = x * x * x := fun x => by noncomm_ring
  -- two-argument rewriting rules for the normal order  hg, l, g, t, d
  have c_l_hg : l * hg = hg * l := hhg_l.symm
  have c_g_hg : g * hg = hg * g := hhg_g.symm
  have c_t_hg : t * hg = hg * t := hhg_t.symm
  have c_d_hg : d * hg = hg * d := hhg_d.symm
  have c_g_l : g * l = l * g := hl_g.symm
  have c_t_l : t * l = l * t := hl_t.symm
  have c_d_l : d * l = l * d := hl_d.symm
  have c_t_g : t * g = g * t := hgc_t.symm
  have c_d_g : d * g = g * d := hgc_d.symm
  have c_d_t : d * t = t * d + 1 := h_comm
  -- the same rules inside a right-associated product
  have m_l_hg : ∀ x : A, l * (hg * x) = hg * (l * x) := fun x => by
    rw [← mul_assoc, c_l_hg, mul_assoc]
  have m_g_hg : ∀ x : A, g * (hg * x) = hg * (g * x) := fun x => by
    rw [← mul_assoc, c_g_hg, mul_assoc]
  have m_t_hg : ∀ x : A, t * (hg * x) = hg * (t * x) := fun x => by
    rw [← mul_assoc, c_t_hg, mul_assoc]
  have m_d_hg : ∀ x : A, d * (hg * x) = hg * (d * x) := fun x => by
    rw [← mul_assoc, c_d_hg, mul_assoc]
  have m_g_l : ∀ x : A, g * (l * x) = l * (g * x) := fun x => by
    rw [← mul_assoc, c_g_l, mul_assoc]
  have m_t_l : ∀ x : A, t * (l * x) = l * (t * x) := fun x => by
    rw [← mul_assoc, c_t_l, mul_assoc]
  have m_d_l : ∀ x : A, d * (l * x) = l * (d * x) := fun x => by
    rw [← mul_assoc, c_d_l, mul_assoc]
  have m_t_g : ∀ x : A, t * (g * x) = g * (t * x) := fun x => by
    rw [← mul_assoc, c_t_g, mul_assoc]
  have m_d_g : ∀ x : A, d * (g * x) = g * (d * x) := fun x => by
    rw [← mul_assoc, c_d_g, mul_assoc]
  have m_d_t : ∀ x : A, d * (t * x) = t * (d * x) + x := fun x => by
    rw [← mul_assoc, c_d_t, add_mul, mul_assoc, one_mul]
  -- the only place where the inverse relation is used inside `hS`
  have hone : hg * g + hg * g = 1 := by rw [← mul_add, ← n2 g]; exact h_inv
  -- clearing the denominator entrywise
  have hS : ∀ k j, S2 t d l g hg k j = T2 t d l g k j * hg := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [S2, T2] <;>
      (try simp only [p2, p3, mul_one, one_mul, sub_eq_add_neg, neg_mul, mul_neg, neg_neg,
        neg_add, mul_add, add_mul, zero_mul, mul_zero, add_zero, zero_add, mul_assoc,
        n2, n2', n3, n3',
        c_l_hg, c_g_hg, c_t_hg, c_d_hg, c_g_l, c_t_l, c_d_l, c_t_g, c_d_g, c_d_t,
        m_l_hg, m_g_hg, m_t_hg, m_d_hg, m_g_l, m_t_l, m_d_l, m_t_g, m_d_g, m_d_t]) <;>
      first
        | noncomm_ring
        -- the `-1` entry of the certificate: this is where `hg = 1/(2 g)` enters
        | rw [← hone]
  -- the polynomial identity, with the denominator cleared
  have hPT : P2 t d l g * T2 t d l g = !![2 * g, 0, 0; 0, 2 * g, 0; 0, 0, 2 * g] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [P2, T2, Matrix.mul_apply, Fin.sum_univ_five] <;>
      (try simp only [p2, p3, mul_one, one_mul, sub_eq_add_neg, neg_mul, mul_neg, neg_neg,
        neg_add, mul_add, add_mul, zero_mul, mul_zero, add_zero, zero_add, mul_assoc,
        n2, n2', n3, n3',
        c_l_hg, c_g_hg, c_t_hg, c_d_hg, c_g_l, c_t_l, c_d_l, c_t_g, c_d_g, c_d_t,
        m_l_hg, m_g_hg, m_t_hg, m_d_hg, m_g_l, m_t_l, m_d_l, m_t_g, m_d_g, m_d_t]) <;>
      first
        | noncomm_ring
        | abel
  ext i j
  have expand : (P2 t d l g * S2 t d l g hg) i j
      = (P2 t d l g * T2 t d l g) i j * hg := by
    simp only [Matrix.mul_apply, hS, Finset.sum_mul, mul_assoc]
  rw [expand, hPT]
  fin_cases i <;> fin_cases j <;> simp [Matrix.one_apply, h_inv']

end GhostTasking
