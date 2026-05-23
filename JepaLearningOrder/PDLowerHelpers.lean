import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.JEPA.Core

/-!
# Helper lemmas for pd_lower_from_offDiag (Lemma B.2)

These helpers establish:
1. Frobenius norm submultiplicativity
2. Frobenius norm positivity for nonzero matrices
3. Gershgorin-style diagonal dominance ⟹ det ≠ 0
4. Eigenvectors from a GenEigenbasis are linearly independent
5. The amplitude matrix factorizes as B = V · Σˣˣ · W̄ · Vᵀ (for det computation)
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

/-! ## Frobenius norm properties -/

/-- Frobenius norm is nonneg. -/
lemma matFrobNorm_nonneg {n m : ℕ} (M : Matrix (Fin n) (Fin m) ℝ) :
    0 ≤ matFrobNorm M :=
  Real.sqrt_nonneg _

/-- Frobenius norm squared equals the sum of squares of entries. -/
lemma matFrobNorm_sq {n m : ℕ} (M : Matrix (Fin n) (Fin m) ℝ) :
    matFrobNorm M ^ 2 = ∑ i, ∑ j, (M i j) ^ 2 := by
  unfold matFrobNorm
  rw [Real.sq_sqrt (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _)]

/-- Frobenius norm is zero iff the matrix is zero. -/
lemma matFrobNorm_eq_zero_iff {n m : ℕ} (M : Matrix (Fin n) (Fin m) ℝ) :
    matFrobNorm M = 0 ↔ M = 0 := by
  constructor
  · intro h
    unfold matFrobNorm at h
    rw [Real.sqrt_eq_zero (Finset.sum_nonneg fun _ _ =>
      Finset.sum_nonneg fun _ _ => sq_nonneg _)] at h
    ext i j
    have := Finset.sum_eq_zero_iff_of_nonneg
      (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) |>.mp h i (Finset.mem_univ _)
    have := Finset.sum_eq_zero_iff_of_nonneg
      (fun _ _ => sq_nonneg _) |>.mp this j (Finset.mem_univ _)
    simpa [sq_eq_zero_iff] using this
  · intro h; simp [matFrobNorm, h]

/-- Frobenius norm is positive for nonzero matrices. -/
lemma matFrobNorm_pos_of_ne_zero {n m : ℕ} (M : Matrix (Fin n) (Fin m) ℝ)
    (h : M ≠ 0) : 0 < matFrobNorm M := by
  rw [lt_iff_le_and_ne]
  exact ⟨matFrobNorm_nonneg _, fun h' => h ((matFrobNorm_eq_zero_iff M).mp h'.symm)⟩

/-
**Frobenius norm submultiplicativity.**
    ‖A * B‖_F ≤ ‖A‖_F * ‖B‖_F.

    Proof: ‖AB‖_F² = ∑ᵢ ∑ⱼ (∑ₖ Aᵢₖ Bₖⱼ)²
           ≤ ∑ᵢ ∑ⱼ (∑ₖ Aᵢₖ²)(∑ₖ Bₖⱼ²)   (Cauchy-Schwarz on each (i,j))
           = (∑ᵢ ∑ₖ Aᵢₖ²)(∑ₖ ∑ⱼ Bₖⱼ²) = ‖A‖_F² · ‖B‖_F².
-/
lemma matFrobNorm_mul_le {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) :
    matFrobNorm (A * B) ≤ matFrobNorm A * matFrobNorm B := by
  -- By Cauchy-Schwarz on each entry: (AB)_{ij} = ∑_k A_{ik} B_{kj}, so |(AB)_{ij}|^2 ≤ (∑_k A_{ik}^2)(∑_k B_{kj}^2).
  have h_cauchy_schwarz : ∀ i j, (∑ k, A i k * B k j)^2 ≤ (∑ k, A i k^2) * (∑ k, B k j^2) := by
    exact?;
  refine Real.sqrt_le_iff.mpr ?_;
  refine' ⟨ mul_nonneg ( Real.sqrt_nonneg _ ) ( Real.sqrt_nonneg _ ), _ ⟩;
  convert Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => h_cauchy_schwarz i j using 1 ; norm_num [ Finset.mul_sum _ _ _, Finset.sum_mul, matFrobNorm ] ; ring;
  norm_num [ ← Finset.mul_sum _ _ _, ← Finset.sum_mul, Real.sq_sqrt ( Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ) ] ; ring;
  exact Or.inl ( Finset.sum_comm )

/-! ## Diagonal dominance (Gershgorin) -/

/-
**Gershgorin diagonal dominance implies det ≠ 0.**
    If a d×d real matrix has diagonal entries with |Bᵣᵣ| ≥ a > 0 and
    off-diagonal |Bᵣₛ| ≤ b with (d-1)·b < a, then B is nonsingular.

    Proof: Suppose Bx = 0 for unit vector x. Let r₀ = argmax |xᵣ|, so |xᵣ₀| ≥ 1/√d.
    From (Bx)ᵣ₀ = 0: |Bᵣ₀ᵣ₀||xᵣ₀| ≤ ∑_{s≠r₀} |Bᵣ₀ₛ||xₛ| ≤ b(d-1)|xᵣ₀|.
    So a ≤ (d-1)b, contradicting hdom.
-/
lemma diag_dom_det_ne_zero {d : ℕ} (B : Matrix (Fin d) (Fin d) ℝ)
    (a : ℝ) (ha : 0 < a) (b : ℝ) (hb : 0 ≤ b)
    (hdiag : ∀ r : Fin d, B r r ≥ a)
    (hoff : ∀ r s : Fin d, r ≠ s → |B r s| ≤ b)
    (hdom : b * ((d : ℝ) - 1) < a) :
    B.det ≠ 0 := by
  rcases d with ( _ | _ | d ) <;> norm_num at *;
  · simp_all +decide [ Fin.eq_zero, Matrix.det_succ_row_zero ];
    linarith;
  · have h_diag_dominant : ∀ r : Fin (d + 2), |B r r| > ∑ s ∈ Finset.univ.erase r, |B r s| := by
      intro r
      have h_sum_off_diag : ∑ s ∈ Finset.univ.erase r, |B r s| ≤ (d + 1) * b := by
        exact le_trans ( Finset.sum_le_sum fun s hs => hoff r s <| by aesop ) <| by norm_num;
      cases abs_cases ( B r r ) <;> linarith [ hdiag r ];
    exact?

/-! ## Eigenbasis linear independence -/

/-
**Eigenvectors from a GenEigenbasis are linearly independent.**
    Proof: Biorthogonality gives dotProduct vᵣ (Σˣˣ vₛ) = δᵣₛ μᵣ.
    If ∑ cᵣ vᵣ = 0, then dotProduct (Σˣˣ vₛ) (∑ cᵣ vᵣ) = cₛ μₛ = 0,
    and μₛ > 0 gives cₛ = 0.
-/
lemma eigvecs_linearIndependent {d : ℕ} (dat : JEPAData d) (eb : GenEigenbasis dat) :
    LinearIndependent ℝ (fun r : Fin d => (eb.pairs r).v) := by
  refine' Fintype.linearIndependent_iff.2 _;
  intro g hg i
  have h_dot : ∑ j, g j • dotProduct (eb.pairs i).v (dat.SigmaXX.mulVec (eb.pairs j).v) = 0 := by
    convert congr_arg ( fun x => dotProduct ( eb.pairs i |> GenEigenpair.v ) ( dat.SigmaXX.mulVec x ) ) hg using 1 <;> simp +decide [ dotProduct, Matrix.mulVec, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ];
    exact?;
  rw [ Finset.sum_eq_single i ] at h_dot;
  · have := eb.hpos i; have := eb.pairs i |>.hrho_pos; have := eb.pairs i |>.hmu_pos; have := eb.pairs i |>.hmu_def; aesop;
  · exact fun j _ hj => by rw [ eb.hbiorthog i j ( Ne.symm hj ), smul_zero ] ;
  · aesop

/-! ## Amplitude matrix det factorization -/

/-
The amplitude matrix B with Bᵣₛ = uᵣᵀ W̄ vₛ satisfies
    det(B) = det(V)² · det(Σˣˣ) · det(W̄)
    where V is the matrix with rows vᵣ.

    Proof: B = V · Σˣˣ · W̄ · Vᵀ (entry-wise verification), then
    det(B) = det(V) · det(Σˣˣ) · det(W̄) · det(Vᵀ) = det(V)² · det(Σˣˣ) · det(W̄).
-/
lemma amplitude_det_factorization {d : ℕ} (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) :
    Matrix.of (fun (r : Fin d) (s : Fin d) =>
        dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs s).v))
    = Matrix.of (fun (r : Fin d) (i : Fin d) => (eb.pairs r).v i)
      * dat.SigmaXX * Wbar
      * (Matrix.of (fun (r : Fin d) (i : Fin d) => (eb.pairs r).v i))ᵀ := by
  ext r s; simp +decide [ dualBasis, Matrix.mul_apply, dotProduct ] ;
  simp +decide only [Matrix.mulVec, dotProduct, Finset.mul_sum _ _ _, Finset.sum_mul _ _ _];
  rw [ Finset.sum_comm ] ; congr ; ext ; congr ; ext ; ring;
  exact Finset.sum_congr rfl fun _ _ => by rw [ show dat.SigmaXX _ _ = dat.SigmaXX _ _ from dat.hSigmaXX_pos.1.apply _ _ ] ; ring;

/-! ## Combining: Wbar det ≠ 0 -/

/-
Under diagonal/off-diagonal amplitude conditions with Gershgorin dominance,
    det(W̄) ≠ 0.
-/
lemma wbar_det_ne_zero {d : ℕ} (hd : 0 < d) (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ)
    (epsilon : ℝ) (heps : 0 < epsilon)
    (L : ℕ) (hL : 2 ≤ L)
    (c_w : ℝ) (hc_w : 0 < c_w)
    (hdiag : ∀ r : Fin d, diagAmplitude dat eb Wbar r ≥ c_w * epsilon ^ ((1 : ℝ) / L))
    (δ : ℝ) (hδ_nn : 0 ≤ δ)
    (hoff : ∀ r s : Fin d, r ≠ s →
        |offDiagAmplitude dat eb Wbar r s| ≤ δ * epsilon ^ ((1 : ℝ) / L))
    (hδ_small : δ * ((d : ℝ) - 1) < c_w) :
    Wbar.det ≠ 0 := by
  -- Define amplitude matrix
  set V := Matrix.of (fun (r : Fin d) (i : Fin d) => (eb.pairs r).v i) with hV_def
  set B := Matrix.of (fun (r : Fin d) (s : Fin d) =>
      dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs s).v)) with hB_def
  -- B has diagonal ≥ c_w ε^{1/L} and off-diagonal ≤ δ ε^{1/L}
  have heps_rpow_pos : 0 < epsilon ^ ((1 : ℝ) / L) := Real.rpow_pos_of_pos heps _
  have hB_diag : ∀ r : Fin d, B r r ≥ c_w * epsilon ^ ((1 : ℝ) / L) := by
    intro r; simp only [hB_def, Matrix.of_apply]; exact hdiag r
  have hB_off : ∀ r s : Fin d, r ≠ s → |B r s| ≤ δ * epsilon ^ ((1 : ℝ) / L) := by
    intro r s hrs; simp only [hB_def, Matrix.of_apply]; exact hoff r s hrs
  -- Diagonal dominance: (δ ε^{1/L}) * (d-1) < c_w ε^{1/L}
  have hdom : δ * epsilon ^ ((1 : ℝ) / L) * ((d : ℝ) - 1) < c_w * epsilon ^ ((1 : ℝ) / L) := by
    nlinarith [mul_comm δ ((d : ℝ) - 1)]
  -- det(B) ≠ 0 by Gershgorin
  have hB_det : B.det ≠ 0 := diag_dom_det_ne_zero B
    (c_w * epsilon ^ ((1 : ℝ) / L)) (by positivity)
    (δ * epsilon ^ ((1 : ℝ) / L)) (by positivity)
    hB_diag hB_off hdom
  -- B = V * SigmaXX * Wbar * V^T
  have hB_eq : B = V * dat.SigmaXX * Wbar * Vᵀ := by
    rw [hB_def, hV_def]; exact amplitude_det_factorization dat eb Wbar
  -- det(V) ≠ 0 from linear independence
  have hV_li := eigvecs_linearIndependent dat eb
  have hV_det : V.det ≠ 0 := by
    rw [ Fintype.linearIndependent_iff ] at hV_li;
    contrapose! hV_li;
    obtain ⟨ g, hg ⟩ := Matrix.exists_vecMul_eq_zero_iff.mpr hV_li;
    exact ⟨ g, by simpa [ funext_iff, Matrix.vecMul, dotProduct ] using hg.2, Function.ne_iff.mp hg.1 ⟩ -- From linear independence of rows of V
  -- det(SigmaXX) > 0 from PosDef
  have hSigmaXX_det : 0 < dat.SigmaXX.det := dat.hSigmaXX_pos.det_pos
  -- B = V * SigmaXX * Wbar * V^T, so det(B) = det(V)^2 * det(SigmaXX) * det(Wbar)
  intro h_contra
  apply hB_det
  rw [hB_eq, Matrix.det_mul, Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose, h_contra]
  ring