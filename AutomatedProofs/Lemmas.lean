import Mathlib

/-!
# Auxiliary Mathematical Lemmas for JEPA

Two classical results needed to complete the sorry'd goals in JEPA.lean.

## 1. Positive-definite Frobenius lower bound
Reference: Horn & Johnson, *Matrix Analysis* (2nd ed.), Theorem 4.2.2 (Rayleigh quotient theorem).

For any positive definite real symmetric matrix A, the minimum Rayleigh quotient
  λ_min(A) = min_{x ≠ 0} xᵀAx / xᵀx
is positive. Consequently, for any matrix M:
  Tr(MᵀMA) ≥ λ_min(A) · Tr(MᵀM) = λ_min(A) · ‖M‖_F².

Proof strategy (Horn-Johnson §4.2):
  Step 1: Spectrally decompose A = UᵀΛU (real spectral theorem, A symmetric → diagonalisable).
  Step 2: For any unit vector x, xᵀAx = Σᵢ λᵢ(Ux)ᵢ² is a convex combination of eigenvalues,
          so xᵀAx ≥ λ_min.
  Step 3: Apply to each column eᵢ of M: (Meᵢ)ᵀ A (Meᵢ) ≥ λ_min · ‖Meᵢ‖².
  Step 4: Sum over i: Tr(MᵀMA) = Σᵢ eᵢᵀMᵀMAeᵢ ≥ λ_min · Σᵢ ‖Meᵢ‖² = λ_min · ‖M‖_F².

## 2. Variable-coefficient Grönwall inequality (integral form)
Reference: Classical ODE theory; see also Teschl *ODE & Dynamical Systems*, used in proof of Thm 2.10
(p.47): |θ(t,x)| ≤ R̃(x)·exp(∫₀ᵀ ‖A(s)‖ ds) from the bound |θ| ≤ R̃ + ∫‖A‖·|θ|.

Standard statement: If u : [0,T] → ℝ is continuous, β : [0,T] → ℝ is non-negative and
integrable, and u(t) ≤ c + ∫₀ᵗ β(s)·u(s) ds for all t ∈ [0,T], then
  u(t) ≤ c · exp(∫₀ᵗ β(s) ds).

Proof strategy:
  Step 1: Let I(t) = ∫₀ᵗ β(s)·u(s) ds. Then I'(t) = β(t)·u(t) ≤ β(t)·(c + I(t)).
  Step 2: So I'(t) - β(t)·I(t) ≤ β(t)·c.
  Step 3: Multiply by integrating factor exp(-∫₀ᵗ β): d/dt[I(t)·exp(-∫₀ᵗ β)] ≤ β(t)·c·exp(-∫₀ᵗ β).
  Step 4: Integrate from 0 to t: I(t)·exp(-∫₀ᵗ β) ≤ c·(1 - exp(-∫₀ᵗ β)).
  Step 5: Therefore I(t) ≤ c·(exp(∫₀ᵗ β) - 1), and u(t) ≤ c + I(t) ≤ c·exp(∫₀ᵗ β). □
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

/-! ## Section 1: Positive-Definite Quadratic Lower Bound -/

set_option maxHeartbeats 400000 in
/-- **Rayleigh quotient lower bound (Horn-Johnson 4.2.2).**
    For a positive definite real symmetric d×d matrix A, there exists λ > 0 such that
    xᵀAx ≥ λ · ‖x‖² for all x : Fin d → ℝ.

    This is the Rayleigh quotient minimum: λ = min_{x ≠ 0} xᵀAx / xᵀx, which is
    positive because A is positive definite and the unit sphere is compact (finite-dimensional).

    PROVIDED SOLUTION
    Step 1: The function f(x) = xᵀAx = dotProduct x (A.mulVec x) is continuous on ℝᵈ.
    Step 2: Restrict to the Euclidean unit sphere S^{d-1} = {x | ‖x‖ = 1}, which is compact
            (Metric.sphere 0 1 is compact in (Fin d → ℝ) by finite-dimensionality).
    Step 3: f attains its minimum λ on S^{d-1}: use IsCompact.exists_isMinOn with continuity of f.
    Step 4: λ > 0: for any x ≠ 0, xᵀAx > 0 by Matrix.PosDef; in particular λ > 0.
    Step 5: For general x: if x = 0, trivial. If x ≠ 0, let y = x/‖x‖ (unit vector),
            then xᵀAx = ‖x‖² · yᵀAy ≥ ‖x‖² · λ = λ · dotProduct x x.
    Key Mathlib API:
    - Matrix.PosDef.pos_of_ne_zero : ∀ x ≠ 0, 0 < dotProduct x (A.mulVec x)
    - Metric.isCompact_sphere : IsCompact (Metric.sphere 0 1)  [for finite-dimensional spaces]
    - IsCompact.exists_isMinOn -/
lemma pd_quadratic_lower_bound {d : ℕ} (hd : 0 < d)
    (A : Matrix (Fin d) (Fin d) ℝ) (hA : A.PosDef) :
    ∃ lam : ℝ, 0 < lam ∧ ∀ x : Fin d → ℝ,
      dotProduct x (A.mulVec x) ≥ lam * dotProduct x x := by
  have h_cont_pos : ∃ lam > 0, ∀ x : Fin d → ℝ, ‖x‖ = 1 → dotProduct x (A.mulVec x) ≥ lam := by
    have h_cont_pos : ContinuousOn (fun x : Fin d → ℝ => dotProduct x (A.mulVec x)) (Metric.sphere (0 : Fin d → ℝ) 1) ∧ ∀ x : Fin d → ℝ, ‖x‖ = 1 → 0 < dotProduct x (A.mulVec x) := by
      refine' ⟨ _, _ ⟩;
      · fun_prop;
      · have := hA.2;
        intro x hx; specialize @this ( Finsupp.equivFunOnFinite.symm x ) ; simp_all +decide [ Matrix.mulVec, dotProduct, Finsupp.sum_fintype ] ;
        by_cases h : Finsupp.equivFunOnFinite.symm x = 0 <;> simp_all +decide [ mul_assoc, Finset.mul_sum _ _ _ ];
        simp_all +decide [ Finsupp.ext_iff, funext_iff ];
        norm_num [ show x = 0 from funext h ] at hx;
    obtain ⟨lam, hl⟩ : ∃ lam ∈ (Set.image (fun x : Fin d → ℝ => dotProduct x (A.mulVec x)) (Metric.sphere (0 : Fin d → ℝ) 1)), ∀ y ∈ (Set.image (fun x : Fin d → ℝ => dotProduct x (A.mulVec x)) (Metric.sphere (0 : Fin d → ℝ) 1)), lam ≤ y := by
      apply_rules [ IsCompact.exists_isLeast, CompactIccSpace.isCompact_Icc ];
      · exact IsCompact.image_of_continuousOn ( isCompact_sphere _ _ ) h_cont_pos.1;
      · rcases d with ( _ | _ | d ) <;> norm_num at *;
    exact ⟨ lam, by obtain ⟨ x, hx, rfl ⟩ := hl.1; exact h_cont_pos.2 x <| by simpa using hx, fun x hx => hl.2 _ <| Set.mem_image_of_mem _ <| by simpa using hx ⟩;
  cases' h_cont_pos with lam hlam;
  have h_general : ∀ x : Fin d → ℝ, x ≠ 0 → dotProduct x (A.mulVec x) ≥ lam * ‖x‖^2 := by
    intro x hx_ne; have := hlam.2 ( ‖x‖⁻¹ • x ) ( by simp +decide [ norm_smul, hx_ne ] ) ; simp_all +decide [ Matrix.mulVec_smul, dotProduct_smul ] ;
    rw [ inv_mul_eq_div, inv_mul_eq_div, div_div, le_div_iff₀ ] at this <;> nlinarith [ norm_pos_iff.mpr hx_ne ];
  refine' ⟨ lam / d, div_pos hlam.1 ( Nat.cast_pos.mpr hd ), fun x => _ ⟩;
  by_cases hx : x = 0 <;> simp_all +decide [ div_mul_eq_mul_div ];
  refine' le_trans _ ( h_general x hx );
  rw [ div_le_iff₀ ( by positivity ) ];
  norm_num [ mul_assoc, dotProduct ];
  exact mul_le_mul_of_nonneg_left ( le_trans ( Finset.sum_le_sum fun _ _ => show x _ * x _ ≤ ‖x‖ ^ 2 by nlinarith only [ abs_le.mp ( norm_le_pi_norm x ‹_› ) ] ) ( by norm_num; nlinarith ) ) hlam.1.le

/-- **Frobenius trace lower bound (consequence of Horn-Johnson 4.2.2).**
    For a positive definite A and any matrix M:
      Tr(MᵀMA) ≥ λ_min(A) · ‖M‖_F²  where  ‖M‖_F² = Tr(MᵀM) = Σᵢ Σⱼ M(i,j)².

    Proof: Apply pd_quadratic_lower_bound column-by-column.
    For each column i: eᵢᵀMᵀMAeᵢ = (Meᵢ)ᵀ A (Meᵢ) ≥ λ · (Meᵢ)ᵀ(Meᵢ).
    Sum: Tr(MᵀMA) = Σᵢ eᵢᵀMᵀMAeᵢ ≥ λ · Σᵢ eᵢᵀMᵀMeᵢ = λ · Tr(MᵀM).

    PROVIDED SOLUTION
    Step 1: Obtain λ from pd_quadratic_lower_bound applied to A.
    Step 2: Unfold Matrix.trace as a sum over the diagonal.
    Step 3: For each index i, the (i,i) entry of MᵀMA is dotProduct (M.col i) (A.mulVec (M.col i)).
            (Use Matrix.mulVec and dotProduct to express column inner products.)
    Step 4: Apply the bound: each term ≥ λ · dotProduct (M.col i) (M.col i).
    Step 5: The sum Σᵢ dotProduct (M.col i) (M.col i) = Tr(MᵀM) = Σᵢ Σⱼ M(j,i)² = ‖M‖_F². -/
lemma frobenius_pd_lower_bound {d : ℕ} (hd : 0 < d)
    (A : Matrix (Fin d) (Fin d) ℝ) (hA : A.PosDef)
    (M : Matrix (Fin d) (Fin d) ℝ) :
    ∃ lam : ℝ, 0 < lam ∧
      Matrix.trace (Mᵀ * M * A) ≥ lam * Matrix.trace (Mᵀ * M) := by
  obtain ⟨lam, hlam_pos, hlam_bound⟩ := pd_quadratic_lower_bound hd A hA
  use lam, hlam_pos
  -- Tr(MᵀMA) = Σᵢ (MᵀMA)ᵢᵢ = Σᵢ dotProduct (M.col i) (A.mulVec (M.col i))
  -- ≥ Σᵢ λ · dotProduct (M.col i) (M.col i) = λ · Tr(MᵀM)
  have h_trace_ineq : Matrix.trace (Mᵀ * M * A) = Matrix.trace (A * Mᵀ * M) := by
    rw [ ← Matrix.trace_mul_comm ] ; simp +decide [ mul_assoc ] ;
  have h_term_ineq : ∀ i, Matrix.trace (A * (Matrix.of (fun j k => M i j * M i k))) ≥ lam * Matrix.trace (Matrix.of (fun j k => M i j * M i k)) := by
    simp_all +decide [ Matrix.trace, Matrix.mul_apply ];
    intro i; specialize hlam_bound ( M i ) ; simp_all +decide [ Matrix.mulVec, dotProduct, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ] ;
  have h_sum_ineq : Matrix.trace (A * Mᵀ * M) = ∑ i, Matrix.trace (A * (Matrix.of (fun j k => M i j * M i k))) := by
    simp +decide [ Matrix.mul_apply, Matrix.trace ];
    exact Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by rw [ Finset.sum_mul _ _ _ ] ; ac_rfl );
  have h_sum_ineq2 : Matrix.trace (Mᵀ * M) = ∑ i, Matrix.trace (Matrix.of (fun j k => M i j * M i k)) := by
    simp +decide [ Matrix.trace, Matrix.mul_apply ];
    exact Finset.sum_comm;
  simpa only [ h_trace_ineq, h_sum_ineq, h_sum_ineq2, Finset.mul_sum _ _ _ ] using Finset.sum_le_sum fun i _ => h_term_ineq i

/-! ## Section 2: Variable-Coefficient Grönwall Inequality -/

/-- **Variable-coefficient Grönwall inequality, integral form.**

    If u : ℝ → ℝ is continuous on [0, T], β : ℝ → ℝ is non-negative and integrable on [0, T],
    and the integral bound u(t) ≤ c + ∫₀ᵗ β(s) · u(s) ds holds for all t ∈ [0, T],
    then u(t) ≤ c · exp(∫₀ᵗ β(s) ds).

    This is the classical Grönwall inequality used in ODE stability analysis.
    Reference: applied in Teschl "ODE and Dynamical Systems" (2012), proof of Theorem 2.10 (p.47).

    PROVIDED SOLUTION
    Step 1: Define I(t) = ∫₀ᵗ β(s) · u(s) ds. Since β and u are integrable, I is differentiable
            with I'(t) = β(t) · u(t) (by intervalIntegral.integral_deriv_right or FTC).
    Step 2: From the hypothesis: u(t) ≤ c + I(t). Multiply the derivative bound:
            I'(t) = β(t) · u(t) ≤ β(t) · (c + I(t)).
    Step 3: So (I(t) + c)' = I'(t) ≤ β(t) · (I(t) + c).
    Step 4: Let w(t) = (I(t) + c) · exp(-∫₀ᵗ β(s) ds).
            Then w'(t) = I'(t) · exp(-∫β) - β(t) · (I(t)+c) · exp(-∫β) ≤ 0.
            So w is non-increasing. Since w(0) = c (as I(0) = 0), we get w(t) ≤ c.
    Step 5: Therefore I(t) + c ≤ c · exp(∫₀ᵗ β), so u(t) ≤ c + I(t) ≤ c · exp(∫₀ᵗ β). □

    Key Mathlib API:
    - intervalIntegral.integral_hasDerivWithinAt_right : FTC for the upper limit
    - Set.Icc.nonneg, MeasureTheory.Integrable
    - Real.add_pow_le_pow_mul_pow_of_sq_le_sq or monotonicity of exp
    - Alternatively: use Mathlib's existing gronwallBound_le for constant coefficients
      by replacing β with its sup (a weaker but sufficient bound for our use case). -/
theorem gronwall_integral_ineq
    {T : ℝ} (hT : 0 ≤ T)
    {u β : ℝ → ℝ} {c : ℝ}
    (hu_cont : ContinuousOn u (Set.Icc 0 T))
    (hβ_cont : ContinuousOn β (Set.Icc 0 T))
    (hβ_nn : ∀ s ∈ Set.Icc 0 T, 0 ≤ β s)
    (hβ_int : IntervalIntegrable β MeasureTheory.volume 0 T)
    (hbound : ∀ t ∈ Set.Icc 0 T,
      u t ≤ c + ∫ s in (0 : ℝ)..t, β s * u s) :
    ∀ t ∈ Set.Icc 0 T,
      u t ≤ c * Real.exp (∫ s in (0 : ℝ)..t, β s) := by
  intro t ht
  have hβu_cont : ContinuousOn (fun s => β s * u s) (Set.Icc 0 T) := hβ_cont.mul hu_cont
  -- Reduce to showing F(t) := c + ∫₀ᵗ βu ≤ c * exp(∫₀ᵗ β)
  have h_uF : u t ≤ c + ∫ s in (0:ℝ)..t, β s * u s := hbound t ht
  suffices h_F : c + ∫ s in (0:ℝ)..t, β s * u s ≤ c * Real.exp (∫ s in (0:ℝ)..t, β s) from
    le_trans h_uF h_F
  -- Define integrating-factor function G(t) = (c + ∫₀ᵗ βu) * exp(-∫₀ᵗ β)
  -- We show G is antitone: G(t) ≤ G(0) = c, then divide by exp(-∫β) > 0
  set B : ℝ → ℝ := fun s => ∫ r in (0:ℝ)..s, β r with hB_def
  set F : ℝ → ℝ := fun s => c + ∫ r in (0:ℝ)..s, β r * u r with hF_def
  set G : ℝ → ℝ := fun s => F s * Real.exp (-B s) with hG_def
  have hG0 : G 0 = c := by
    simp [hG_def, hF_def, hB_def, intervalIntegral.integral_same]
  -- Continuity of B and F (needed for G continuous and for FTC)
  have hB_cont : ContinuousOn B (Set.Icc 0 T) := by
    apply (intervalIntegral.continuousOn_primitive_interval' hβ_int Set.left_mem_uIcc).mono
    rw [Set.uIcc_of_le hT]
  have hF_cont : ContinuousOn F (Set.Icc 0 T) := by
    apply ContinuousOn.add continuousOn_const
    apply (intervalIntegral.continuousOn_primitive_interval'
      (hβu_cont.intervalIntegrable_of_Icc hT) Set.left_mem_uIcc).mono
    rw [Set.uIcc_of_le hT]
  -- G is continuous on [0,T]
  have hG_cont : ContinuousOn G (Set.Icc 0 T) :=
    hF_cont.mul (hB_cont.neg.rexp)
  -- Helper: produce StronglyMeasurableAtFilter from ContinuousOn on Icc,
  -- by restricting to the open Ioo which is a subset
  have smaf_β : ∀ s ∈ Set.Ioo 0 T, StronglyMeasurableAtFilter β (nhds s) MeasureTheory.volume :=
    ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo (hβ_cont.mono Set.Ioo_subset_Icc_self)
  have smaf_βu : ∀ s ∈ Set.Ioo 0 T,
      StronglyMeasurableAtFilter (fun r => β r * u r) (nhds s) MeasureTheory.volume :=
    ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo (hβu_cont.mono Set.Ioo_subset_Icc_self)
  -- G is antitone on [0,T]: G' ≤ 0 follows from u ≤ F and β ≥ 0
  have hG_anti : AntitoneOn G (Set.Icc 0 T) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc 0 T) hG_cont
    · -- DifferentiableOn G (interior [0,T])
      intro s hs
      rw [interior_Icc] at hs
      have hβs_int : IntervalIntegrable β MeasureTheory.volume 0 s :=
        (hβ_cont.mono (Set.Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβus_int : IntervalIntegrable (fun r => β r * u r) MeasureTheory.volume 0 s :=
        (hβu_cont.mono (Set.Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβs_cat : ContinuousAt β s :=
        hβ_cont.continuousAt (Icc_mem_nhds hs.1 hs.2)
      have hβus_cat : ContinuousAt (fun r => β r * u r) s :=
        hβu_cont.continuousAt (Icc_mem_nhds hs.1 hs.2)
      have hB_da : HasDerivAt B (β s) s :=
        intervalIntegral.integral_hasDerivAt_right hβs_int (smaf_β s hs) hβs_cat
      have hI_da : HasDerivAt (fun x => ∫ r in (0:ℝ)..x, β r * u r) (β s * u s) s :=
        intervalIntegral.integral_hasDerivAt_right hβus_int (smaf_βu s hs) hβus_cat
      have hF_da : HasDerivAt F (β s * u s) s := by
        have h := (hasDerivAt_const s c).add hI_da
        simp only [zero_add] at h; exact h
      have hEB_da : HasDerivAt (fun r => Real.exp (-B r)) (Real.exp (-B s) * (-β s)) s :=
        hB_da.neg.exp
      exact (hF_da.mul hEB_da).differentiableAt.differentiableWithinAt
    · -- deriv G ≤ 0 on interior
      intro s hs
      rw [interior_Icc] at hs
      have hs_mem : s ∈ Set.Icc 0 T := ⟨hs.1.le, hs.2.le⟩
      have hβs_int : IntervalIntegrable β MeasureTheory.volume 0 s :=
        (hβ_cont.mono (Set.Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβus_int : IntervalIntegrable (fun r => β r * u r) MeasureTheory.volume 0 s :=
        (hβu_cont.mono (Set.Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβs_cat : ContinuousAt β s :=
        hβ_cont.continuousAt (Icc_mem_nhds hs.1 hs.2)
      have hβus_cat : ContinuousAt (fun r => β r * u r) s :=
        hβu_cont.continuousAt (Icc_mem_nhds hs.1 hs.2)
      have hB_da : HasDerivAt B (β s) s :=
        intervalIntegral.integral_hasDerivAt_right hβs_int (smaf_β s hs) hβs_cat
      have hI_da : HasDerivAt (fun x => ∫ r in (0:ℝ)..x, β r * u r) (β s * u s) s :=
        intervalIntegral.integral_hasDerivAt_right hβus_int (smaf_βu s hs) hβus_cat
      have hF_da : HasDerivAt F (β s * u s) s := by
        have h := (hasDerivAt_const s c).add hI_da
        simp only [zero_add] at h; exact h
      have hEB_da : HasDerivAt (fun r => Real.exp (-B r)) (Real.exp (-B s) * (-β s)) s :=
        hB_da.neg.exp
      have hG_da : HasDerivAt G
          (β s * u s * Real.exp (-B s) + F s * (Real.exp (-B s) * (-β s))) s :=
        hF_da.mul hEB_da
      rw [hG_da.deriv]
      have hus_le_Fs : u s ≤ F s := hbound s hs_mem
      have hβs_nn : 0 ≤ β s := hβ_nn s hs_mem
      have hEB_pos : 0 < Real.exp (-B s) := Real.exp_pos _
      have hrw : β s * u s * Real.exp (-B s) + F s * (Real.exp (-B s) * (-β s)) =
             β s * (u s - F s) * Real.exp (-B s) := by ring
      rw [hrw]
      apply mul_nonpos_of_nonpos_of_nonneg
      · exact mul_nonpos_of_nonneg_of_nonpos hβs_nn (by linarith)
      · exact hEB_pos.le
  -- G(t) ≤ G(0) = c
  have hGt : G t ≤ c := hG0 ▸ hG_anti (Set.left_mem_Icc.mpr hT) ht ht.1
  -- Conclude: c + ∫₀ᵗ βu = F(t) ≤ c * exp(∫₀ᵗ β)
  have hEB_pos : 0 < Real.exp (-B t) := Real.exp_pos _
  have hEB_ne : Real.exp (-B t) ≠ 0 := ne_of_gt hEB_pos
  -- We show: F(t) ≤ c * exp(∫β) by:
  --   F(t) * exp(-∫β) = G(t) ≤ c, so F(t) ≤ c / exp(-∫β) = c * exp(∫β)
  suffices h : F t ≤ c * Real.exp (∫ s in (0:ℝ)..t, β s) from h
  have hGt_Ft : G t = F t * Real.exp (-B t) := rfl
  have hFt_le : F t ≤ c / Real.exp (-B t) := by
    rw [le_div_iff₀ hEB_pos]
    exact hGt_Ft ▸ hGt
  calc F t ≤ c / Real.exp (-B t) := hFt_le
    _ = c * Real.exp (∫ s in (0:ℝ)..t, β s) := by
        rw [hB_def, Real.exp_neg, div_inv_eq_mul]

/-- **Corollary: Grönwall bound for approximate linear ODE.**

    If f : ℝ → ℝ satisfies:
    - |f(t)| ≤ f₀ for some f₀ ≥ 0 (initial bound),
    - |f'(t) + α(t)·f(t)| ≤ η (approximate ODE with α(t) ≥ 0),
    - ∫₀ᵗ α(s) ds ≤ A_int (integral of coefficient bounded),
    then for all t ∈ [0, T]:
      |f(t)| ≤ (f₀ + T · η) · exp(A_int).

    This is the core bound used in offDiag_bound (Theorem 7.3 in JEPA.lean):
    with f = c_{rs}, α = P_{rs}·κ, η = C·ε^{(2L-1)/L}, and A_int = C_int from Lemma 7.2.

    PROVIDED SOLUTION
    Step 1: From |f'(t) + α(t)·f(t)| ≤ η, we get |f'(t)| ≤ α(t)|f(t)| + η.
    Step 2: From FTC: |f(t)| ≤ |f(0)| + ∫₀ᵗ |f'(s)| ds ≤ f₀ + ∫₀ᵗ (α(s)|f(s)| + η) ds.
    Step 3: So |f(t)| ≤ (f₀ + T·η) + ∫₀ᵗ α(s)·|f(s)| ds.
    Step 4: Apply gronwall_integral_ineq with u = |f|, β = α, c = f₀ + T·η.
    Step 5: Conclude |f(t)| ≤ (f₀ + T·η) · exp(∫₀ᵗ α) ≤ (f₀ + T·η) · exp(A_int). -/
lemma gronwall_approx_ode_bound
    {T : ℝ} (hT : 0 < T)
    {f α : ℝ → ℝ} {f₀ η A_int : ℝ}
    (hf₀ : 0 ≤ f₀) (hη : 0 ≤ η) (hA : 0 ≤ A_int)
    (hf_cont : ContinuousOn f (Set.Icc 0 T))
    (hf_deriv : ∀ t ∈ Set.Icc 0 T,
      ∃ f' : ℝ, HasDerivAt f f' t ∧ |f' + α t * f t| ≤ η)
    (hα_nn : ∀ t ∈ Set.Icc 0 T, 0 ≤ α t)
    (hα_int_bound : ∀ t ∈ Set.Icc 0 T,
      ∫ s in (0 : ℝ)..t, α s ≤ A_int)
    (hinit : |f 0| ≤ f₀) :
    ∀ t ∈ Set.Icc 0 T,
      |f t| ≤ (f₀ + T * η) * Real.exp A_int := by
  have h_liminf : ∀ x ∈ Set.Ico (0 : ℝ) T, ∀ r : ℝ, (η : ℝ) < r →
      ∃ᶠ z in nhdsWithin x (Set.Ioi x), (z - x)⁻¹ * (|f z| - |f x|) < r := by
    intro x hx r hr;
    obtain ⟨ f', hf', hf'' ⟩ := hf_deriv x ⟨ hx.1, hx.2.le ⟩;
    by_cases hfx_pos : f x > 0;
    · have h_lim : Filter.Tendsto (fun z => (z - x)⁻¹ * (f z - f x)) (nhdsWithin x (Set.Ioi x)) (nhds f') := by
        rw [ hasDerivAt_iff_tendsto_slope ] at hf';
        simpa [ div_eq_inv_mul ] using hf'.mono_left ( nhdsWithin_mono _ <| by simp +decide );
      have h_lim_abs : Filter.Tendsto (fun z => (z - x)⁻¹ * (|f z| - |f x|)) (nhdsWithin x (Set.Ioi x)) (nhds f') := by
        refine' h_lim.congr' _;
        rw [ Filter.EventuallyEq, eventually_nhdsWithin_iff ];
        filter_upwards [ hf'.continuousAt.eventually ( lt_mem_nhds hfx_pos ) ] with y hy hy' using by rw [ abs_of_pos hy, abs_of_pos hfx_pos ] ;
      exact h_lim_abs.eventually ( gt_mem_nhds <| by nlinarith [ abs_le.mp hf'', hα_nn x ⟨ hx.1, hx.2.le ⟩ ] ) |> fun h => h.frequently;
    · by_cases hfx_neg : f x < 0;
      · have h_lim : Filter.Tendsto (fun z => (z - x)⁻¹ * (|f z| - |f x|)) (nhdsWithin x (Set.Ioi x)) (nhds (-f')) := by
          have h_lim : Filter.Tendsto (fun z => (z - x)⁻¹ * (f z - f x)) (nhdsWithin x (Set.Ioi x)) (nhds f') := by
            rw [ hasDerivAt_iff_tendsto_slope ] at hf';
            simpa [ div_eq_inv_mul ] using hf'.mono_left ( nhdsWithin_mono _ <| by simp +decide );
          have h_lim : ∀ᶠ z in nhdsWithin x (Set.Ioi x), f z < 0 := by
            have := hf'.continuousAt.tendsto;
            exact this.eventually ( gt_mem_nhds hfx_neg ) |> fun h => h.filter_mono nhdsWithin_le_nhds;
          exact Filter.Tendsto.congr' ( by filter_upwards [ h_lim ] with z hz; rw [ abs_of_neg hz, abs_of_neg hfx_neg ] ; ring ) ( ‹Filter.Tendsto ( fun z => ( z - x ) ⁻¹ * ( f z - f x ) ) ( nhdsWithin x ( Set.Ioi x ) ) ( nhds f' ) ›.neg );
        exact h_lim.eventually ( gt_mem_nhds <| by nlinarith [ abs_le.mp hf'', hα_nn x ⟨ hx.1, hx.2.le ⟩ ] ) |> fun h => h.frequently;
      · have h_lim : Filter.Tendsto (fun z => (z - x)⁻¹ * |f z|) (nhdsWithin x (Set.Ioi x)) (nhds (|f'|)) := by
          have h_lim : Filter.Tendsto (fun z => (z - x)⁻¹ * f z) (nhdsWithin x (Set.Ioi x)) (nhds f') := by
            rw [ hasDerivAt_iff_tendsto_slope ] at hf';
            convert hf'.mono_left <| nhdsWithin_mono _ _ using 2 <;> norm_num [ div_eq_inv_mul, slope_def_field ];
            exact Or.inl ( by linarith );
          have := h_lim.abs;
          refine' this.congr' ( by filter_upwards [ self_mem_nhdsWithin ] with z hz using by rw [ abs_mul, abs_inv, abs_of_nonneg ( sub_nonneg.mpr hz.out.le ) ] );
        simp_all +decide [ show f x = 0 by linarith ];
        exact h_lim.eventually ( gt_mem_nhds <| by linarith ) |> fun h => h.frequently
  have h_gronwall : ∀ t ∈ Set.Icc (0 : ℝ) T,
      |f t| ≤ gronwallBound f₀ 0 η (t - 0) := by
    intro t ht
    exact le_gronwallBound_of_liminf_deriv_right_le
      (hf_cont.norm) h_liminf hinit (fun x _ => by simp) t ht
  intro t ht
  have h1 := h_gronwall t ht
  rw [gronwallBound_K0] at h1
  simp only [sub_zero] at h1
  calc |f t| ≤ f₀ + η * t := h1
    _ ≤ f₀ + T * η := by nlinarith [ht.2]
    _ ≤ (f₀ + T * η) * Real.exp A_int := by
        have h_exp := Real.one_le_exp hA
        have h_nn : 0 ≤ f₀ + T * η := by positivity
        nlinarith [mul_le_mul_of_nonneg_left h_exp h_nn]
