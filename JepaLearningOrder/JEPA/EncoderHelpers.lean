import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.OffDiagHelpers
import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic

/-!
# JEPA — Encoder Helpers (Sections 5.4, 5.5)

Frobenius-norm helpers + Phase-A frozen-encoder convergence machinery.
FrobeniusHelpers and EncoderConvergence are kept merged here per the audit
(11-edge bond — splitting them would create 11 new import-line dependencies).
Extracted from `JepaLearningOrder/JEPA.lean` (session 95 split).
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ} (hd : 0 < d)


/-! ## Section 6.5: Bootstrap Consistency
    **Proved in `BootstrapLemmas.lean`** — see `bootstrap_consistency` there.
    The proof assembles three sub-lemmas (Lemmas B.1–B.3):
    - B.1 `offDiag_ftc`: off-diagonal bound via FTC (no bootstrap).
    - B.2 `pd_lower_from_offDiag`: PD lower bound from Gershgorin (Aristotle 53f7f1b1).
    - B.3 `tracking_bound_from_gronwall`: tracking bound via contractive Gronwall.
    The old Picard-Lindelöf continuation argument is bypassed: FTC gives the off-diagonal
    bound directly, and contractive Gronwall closes the tracking argument. -/

/-! ## Section 5.4: Contraction ODE Structure -/

/-! ### Helper lemmas for contraction_ode_structure -/

/-
Cauchy–Schwarz inequality for the Frobenius inner product.
-/
lemma cauchy_schwarz_frob (A B : Matrix (Fin d) (Fin d) ℝ) :
    |∑ i, ∑ j, A i j * B i j| ≤ matFrobNorm A * matFrobNorm B := by
  -- Apply the Cauchy-Schwarz inequality to the inner sum.
  have h_cauchy_schwarz : ∀ (u v : Fin d × Fin d → ℝ), abs (∑ i, u i * v i) ≤ Real.sqrt (∑ i, u i ^ 2) * Real.sqrt (∑ i, v i ^ 2) := by
    intros u v; rw [ ← Real.sqrt_mul <| Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ; exact Real.abs_le_sqrt <| by exact?;
  convert h_cauchy_schwarz ( fun p => A p.1 p.2 ) ( fun p => B p.1 p.2 ) using 1;
  · erw [ Finset.sum_product ];
  · unfold matFrobNorm;
    erw [ Finset.sum_product, Finset.sum_product ]

/-
HasDerivAt for the sum of squares of matrix entries.
-/
lemma hasDerivAt_sum_sq
    (F : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (F'_t : Matrix (Fin d) (Fin d) ℝ) (t : ℝ)
    (hF : HasDerivAt F F'_t t) :
    HasDerivAt (fun s => ∑ i, ∑ j, (F s i j) ^ 2)
      (∑ i, ∑ j, 2 * F t i j * F'_t i j) t := by
  convert HasDerivAt.sum fun i _ => HasDerivAt.sum fun j _ => ?_ using 1;
  rotate_left;
  use fun i j s => F s i j ^ 2;
  · have h_deriv : HasDerivAt (fun s => F s i j) (F'_t i j) t := by
      convert ( hasDerivAt_pi.mp ( hasDerivAt_pi.mp hF i ) ) j using 1;
    simpa using h_deriv.pow 2;
  · aesop

/-
HasDerivAt for matFrobNorm when the matrix is nonzero.
    Uses chain rule: matFrobNorm = sqrt ∘ (sum of squares),
    and sqrt is differentiable when its argument is nonzero.
-/
lemma hasDerivAt_matFrobNorm_of_ne_zero
    (F : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (F'_t : Matrix (Fin d) (Fin d) ℝ) (t : ℝ)
    (hF : HasDerivAt F F'_t t) (hF_ne : F t ≠ 0) :
    HasDerivAt (fun s => matFrobNorm (F s))
      ((∑ i, ∑ j, F t i j * F'_t i j) / matFrobNorm (F t)) t := by
  have h_chain : HasDerivAt (fun s => ∑ i, ∑ j, (F s i j) ^ 2) (∑ i, ∑ j, (2 * F t i j * F'_t i j)) t := by
    exact?;
  convert HasDerivAt.sqrt h_chain _ using 1;
  · simp +decide [ ← Finset.mul_sum _ _ _, mul_assoc, mul_div_mul_left, div_div, matFrobNorm ];
  · exact fun h => hF_ne <| Matrix.ext fun i j => sq_eq_zero_iff.mp <| by contrapose! h; exact ne_of_gt <| lt_of_lt_of_le ( by exact lt_of_le_of_ne ( sq_nonneg _ ) ( Ne.symm h ) ) ( Finset.single_le_sum ( fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg ( F t i j ) ) ( Finset.mem_univ i ) |> le_trans ( Finset.single_le_sum ( fun j _ => sq_nonneg ( F t i j ) ) ( Finset.mem_univ j ) ) ) ;

/-
A matrix A satisfying ‖M*A‖_F ≥ c*‖M‖_F for all M with c > 0 is invertible.
-/
private lemma matrix_isUnit_det_of_frob_lower_bound
    (A : Matrix (Fin d) (Fin d) ℝ)
    (c : ℝ) (hc : 0 < c)
    (h : ∀ M : Matrix (Fin d) (Fin d) ℝ, matFrobNorm (M * A) ≥ c * matFrobNorm M) :
    IsUnit A.det := by
  contrapose! h; simp_all +decide [ ← Matrix.exists_vecMul_eq_zero_iff ] ;
  obtain ⟨ v, hv, hv' ⟩ := h; use Matrix.of ( fun i j => v j ) ; simp_all +decide [ matFrobNorm ] ;
  simp_all +decide [ funext_iff, Matrix.mul_apply ];
  simp_all +decide [ Matrix.vecMul, dotProduct ];
  exact mul_pos ( Real.sqrt_pos.mpr ( Nat.cast_pos.mpr ( Nat.pos_of_ne_zero ( by aesop_cat ) ) ) ) ( Real.sqrt_pos.mpr ( lt_of_lt_of_le ( sq_pos_of_ne_zero ( hv.choose_spec ) ) ( Finset.single_le_sum ( fun i _ => sq_nonneg ( v i ) ) ( Finset.mem_univ _ ) ) ) )

/-
The quasi-static decoder satisfies V_qs * A = B when A is invertible.
-/
private lemma quasiStatic_mul_cancel (dat : JEPAData d)
    (W : Matrix (Fin d) (Fin d) ℝ)
    (hA_inv : IsUnit (W * dat.SigmaXX * Wᵀ).det) :
    (W * dat.SigmaYX * Wᵀ * (W * dat.SigmaXX * Wᵀ)⁻¹) *
      (W * dat.SigmaXX * Wᵀ) =
    W * dat.SigmaYX * Wᵀ := by
  simp_all +decide [ Matrix.isUnit_iff_isUnit_det ]

/-
W̄ Σˣˣ W̄ᵀ is PosDef when the Frobenius lower bound holds.
-/
lemma wbarSigma_posDef (dat : JEPAData d)
    (W : Matrix (Fin d) (Fin d) ℝ)
    (c : ℝ) (hc : 0 < c)
    (h : ∀ M : Matrix (Fin d) (Fin d) ℝ,
      matFrobNorm (M * (W * dat.SigmaXX * Wᵀ)) ≥ c * matFrobNorm M) :
    (W * dat.SigmaXX * Wᵀ).PosDef := by
  -- By definition of $A$, we know that $A$ is invertible.
  have hA_inv : IsUnit (W * dat.SigmaXX * Wᵀ).det := by
    exact?;
  constructor;
  · simp +decide [ Matrix.IsHermitian, Matrix.mul_assoc ];
    have := dat.hSigmaXX_pos.1; simp_all +decide [ Matrix.IsHermitian ] ;
  · intro x hx_ne_zero
    have h_pos : 0 < dotProduct (Wᵀ.mulVec x) (dat.SigmaXX.mulVec (Wᵀ.mulVec x)) := by
      have h_pos : ∀ v : Fin d → ℝ, v ≠ 0 → 0 < dotProduct v (dat.SigmaXX.mulVec v) := by
        have := dat.hSigmaXX_pos.2;
        simp_all +decide [ Matrix.mulVec, dotProduct, Finsupp.sum_fintype ];
        exact fun v hv => by simpa only [ mul_assoc, Finset.mul_sum _ _ _ ] using this ( show Finsupp.equivFunOnFinite.symm v ≠ 0 from by simpa [ Finsupp.ext_iff, funext_iff ] using hv ) ;
      apply h_pos; intro h_zero; simp_all +decide [ Matrix.mulVec ] ;
      exact hx_ne_zero ( by simpa [ hA_inv ] using Matrix.eq_zero_of_mulVec_eq_zero ( show Wᵀ.det ≠ 0 from by simpa [ Matrix.det_transpose ] using hA_inv.1.1 ) h_zero );
    simp_all +decide [ Matrix.mul_assoc, Matrix.dotProduct_mulVec, Matrix.vecMul_mulVec ];
    convert h_pos using 1;
    simp +decide [ Matrix.vecMul, dotProduct, Finsupp.sum_fintype ];
    simp +decide only [mul_assoc, Finset.sum_mul _ _ _];
    exact Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring )

/-
The Frobenius contraction bound: for PD A satisfying the Frobenius
    lower bound, the Frobenius inner product ∑ij M_ij * (MA)_ij is bounded below.
    Requires A to be PosDef (ensures the quadratic form is positive).
-/
private lemma frob_contraction_bound
    (A : Matrix (Fin d) (Fin d) ℝ) (hA : A.PosDef)
    (c : ℝ) (hc : 0 < c)
    (h : ∀ M : Matrix (Fin d) (Fin d) ℝ,
      matFrobNorm (M * A) ≥ c * matFrobNorm M) :
    ∃ lam : ℝ, 0 < lam ∧
      ∀ M : Matrix (Fin d) (Fin d) ℝ,
        ∑ i, ∑ j, M i j * (M * A) i j ≥ lam * ∑ i, ∑ j, (M i j) ^ 2 := by
  have := @pd_quadratic_lower_bound d;
  rcases d with ( _ | d ) <;> simp_all +decide [ dotProduct, sq ];
  · exact ⟨ 1, by norm_num ⟩;
  · obtain ⟨ lam, hl_pos, hl ⟩ := this A hA; use lam; refine' ⟨ hl_pos, fun M => _ ⟩ ; simp_all +decide [ Matrix.mulVec, dotProduct, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ] ;
    convert Finset.sum_le_sum fun i _ => hl ( fun j => M i j ) using 1 ; simp +decide [ Matrix.mul_apply, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ];
    exact Finset.sum_congr rfl fun _ _ => Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring )

/-
Uniform Frobenius contraction bound. For each t ∈ Icc 0 t_max, the Frobenius
    inner product ∑ij M_ij * (M * A(t))_ij is bounded below by a UNIFORM constant
    times ∑ij M_ij². The uniformity follows because pd_quadratic_lower_bound's lam
    depends on A only through the minimum on the compact unit sphere, and from hPD
    this minimum is at least c₀ * eps_coeff (using PosDef + Frobenius lower bound).

The gradient of the decoder loss equals ΔV * A when A is invertible.
    gradV dat W V = V*A - B and V_qs*A = B, so gradV = (V - V_qs)*A.
-/
lemma gradV_eq_delta_mul_A (dat : JEPAData d)
    (W V_val : Matrix (Fin d) (Fin d) ℝ)
    (hA_inv : IsUnit (W * dat.SigmaXX * Wᵀ).det) :
    gradV dat W V_val =
      (V_val - quasiStaticDecoder dat W) * (W * dat.SigmaXX * Wᵀ) := by
  simp +decide only [gradV, quasiStaticDecoder];
  simp +decide [ sub_mul, mul_assoc, hA_inv ];
  simp_all +decide [ mul_assoc, Matrix.isUnit_iff_isUnit_det ]

set_option maxHeartbeats 800000 in
lemma uniform_frob_contraction (dat : JEPAData d)
    (Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (c₀ : ℝ) (hc₀ : 0 < c₀) (eps_coeff : ℝ) (heps_coeff : 0 < eps_coeff)
    (t_max : ℝ)
    (hPD : ∀ t ∈ Set.Icc 0 t_max, ∀ M : Matrix (Fin d) (Fin d) ℝ,
      matFrobNorm (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ≥ c₀ * eps_coeff * matFrobNorm M) :
    ∃ lam : ℝ, 0 < lam ∧ ∀ t ∈ Set.Icc 0 t_max,
      ∀ M : Matrix (Fin d) (Fin d) ℝ,
        ∑ i, ∑ j, M i j * (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) i j ≥
          lam * ∑ i, ∑ j, (M i j) ^ 2 := by
  have hPD_symm : ∀ t ∈ Set.Icc 0 t_max, ∀ v : Fin d → ℝ, dotProduct ( (Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec v ) ( (Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec v ) ≥ (c₀ * eps_coeff) ^ 2 * dotProduct v v := by
    intro t ht v;
    have hPD_symm : ∀ i : Fin d, matFrobNorm (Matrix.of (fun j k => if j = i then v k else 0) * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ≥ c₀ * eps_coeff * matFrobNorm (Matrix.of (fun j k => if j = i then v k else 0)) := by
      exact fun i => hPD t ht _;
    have hPD_symm_sq : ∀ i : Fin d, ∑ j, ((Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec v) j ^ 2 ≥ (c₀ * eps_coeff) ^ 2 * ∑ j, v j ^ 2 := by
      intro i
      specialize hPD_symm i
      have hPD_symm_sq_i : (∑ j, ((Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec v) j ^ 2) ≥ (c₀ * eps_coeff) ^ 2 * (∑ j, v j ^ 2) := by
        have hPD_symm_sq_i : matFrobNorm (Matrix.of (fun j k => if j = i then v k else 0) * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ^ 2 ≥ (c₀ * eps_coeff) ^ 2 * matFrobNorm (Matrix.of (fun j k => if j = i then v k else 0)) ^ 2 := by
          simpa only [ mul_pow ] using pow_le_pow_left₀ ( mul_nonneg ( mul_nonneg hc₀.le heps_coeff.le ) ( Real.sqrt_nonneg _ ) ) hPD_symm 2
        convert hPD_symm_sq_i using 1 <;> norm_num [ matFrobNorm ];
        · rw [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ; simp +decide [ Matrix.mul_apply, Matrix.mulVec, dotProduct, Finset.mul_sum _ _ _, Finset.sum_mul _ _ _, mul_assoc, mul_comm, mul_left_comm, sq ] ; ring;
          refine' Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => _;
          have := dat.hSigmaXX_pos.1; simp_all +decide [ Matrix.IsSymm, Matrix.mul_apply, Matrix.mulVec, dotProduct ] ;
          simp_all +decide [ Matrix.IsHermitian, Matrix.mul_apply, Matrix.mulVec, dotProduct ];
          rw [ ← Matrix.ext_iff ] at this ; aesop;
        · exact Or.inl <| by rw [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ;
      exact hPD_symm_sq_i;
    cases d <;> simp_all +decide [ dotProduct ];
    simpa only [ sq ] using hPD_symm_sq;
  have hPD_symm : ∀ t ∈ Set.Icc 0 t_max, ∀ v : Fin d → ℝ, dotProduct v ( (Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec v ) ≥ (c₀ * eps_coeff) * dotProduct v v := by
    intros t ht v
    apply pd_quadratic_from_norm_bound (Wbar t * dat.SigmaXX * (Wbar t)ᵀ) (by
    apply wbarSigma_posDef dat (Wbar t) (c₀ * eps_coeff) (by
    positivity) (by
    exact hPD t ht)) (c₀ * eps_coeff) (by
    positivity) (by
    exact hPD_symm t ht) v;
  refine' ⟨ c₀ * eps_coeff, mul_pos hc₀ heps_coeff, fun t ht M => _ ⟩;
  have h_sum : ∑ i, ∑ j, M i j * (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) i j = ∑ i, dotProduct (M i) ((Wbar t * dat.SigmaXX * (Wbar t)ᵀ).mulVec (M i)) := by
    simp +decide [ Matrix.mulVec, dotProduct, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ];
    simp +decide [ Matrix.mul_apply, Finset.mul_sum _ _ _ ];
    refine' Finset.sum_congr rfl fun i hi => Finset.sum_comm.trans ( Finset.sum_congr rfl fun j hj => Finset.sum_congr rfl fun k hk => _ );
    ac_rfl;
  rw [ h_sum, Finset.mul_sum _ _ _ ];
  exact Finset.sum_le_sum fun i _ => by simpa [ sq, dotProduct ] using hPD_symm t ht ( M i ) ;

/-
**Lemma (Contraction ODE structure).**
    Under the JEPA decoder gradient flow, with the encoder Frobenius–PD lower bound
    `‖M · (W̄ Σˣˣ W̄ᵀ)‖_F ≥ c₀ ε^{2/L} ‖M‖_F` and V_qs drift bounded by D₀ ε², the
    tracking error f(t) = ‖V(t) − V_qs(W̄(t))‖_F satisfies the contractive ODE

        f'(t) ≤ −(c₀ ε^{2/L}) f(t) + D₀ ε²

    for uniform constants c₀, D₀ > 0, independent of ε and t.

    Requires the tracking error to be nonzero, since matFrobNorm = √(∑ squares) is not
    differentiable at 0 when the derivative of the matrix function is nonzero (the function
    has a V-shaped kink). In the physical setting this holds since the decoder has not
    perfectly converged to the quasi-static value at any finite time.

    Once proved, this discharges hypothesis (R2) of `JEPA_rho_ordering`, removing it from
    the theorem's signature in favour of `hVqs_deriv_exists`, `hDrift_bound`, and `hPD_lower`.

    PROOF OUTLINE
    Step 1: ΔV̇ = −ΔV · A − Ḋ from the ODE and V_qs · A = B.
    Step 2: HasDerivAt for f(t) via chain rule for sqrt ∘ (∑ squares).
    Step 3: Contraction bound from hPD_lower and frobenius_pd_lower_bound.
    Step 4: Drift bound from Cauchy–Schwarz and hDrift_bound.
    Step 5: Combine.
-/
lemma contraction_ode_structure {d : ℕ} (hd : 0 < d) (dat : JEPAData d)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    -- Decoder satisfies the JEPA gradient-flow ODE
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt V (-(gradV dat (Wbar t) (V t))) t)
    -- V_qs ∘ Wbar is differentiable on (0, t_max)
    (hVqs_deriv_exists : ∀ t ∈ Set.Ico 0 t_max,
        ∃ Vqs_d : Matrix (Fin d) (Fin d) ℝ,
          HasDerivAt (fun s => quasiStaticDecoder dat (Wbar s)) Vqs_d t)
    -- Drift bound: ‖d/dt V_qs(W̄(t))‖_F ≤ D₀ ε² (follows from hWbar_slow + chain rule)
    (hDrift_bound : ∃ D₀ : ℝ, 0 < D₀ ∧ ∀ t ∈ Set.Ico 0 t_max,
        matFrobNorm (deriv (fun s => quasiStaticDecoder dat (Wbar s)) t) ≤ D₀ * epsilon ^ 2)
    -- Frobenius PD lower bound on W̄(t) Σˣˣ W̄(t)ᵀ (derivable from balanced init + hoff_small)
    (hPD_lower : ∃ c₀ : ℝ, 0 < c₀ ∧ ∀ t ∈ Set.Icc 0 t_max,
        ∀ M : Matrix (Fin d) (Fin d) ℝ,
          matFrobNorm (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ≥
            c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M)
    -- Tracking error is nonzero (needed for differentiability of matFrobNorm at 0)
    (hDelta_nz : ∀ t ∈ Set.Ico 0 t_max,
        V t - quasiStaticDecoder dat (Wbar t) ≠ 0)
    : ∃ (c₀ D₀ : ℝ), 0 < c₀ ∧ 0 < D₀ ∧
      ∀ t ∈ Set.Ico 0 t_max,
        ∃ f' : ℝ,
          HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat (Wbar s))) f' t ∧
          f' ≤ -(c₀ * epsilon ^ ((2 : ℝ) / L)) *
                matFrobNorm (V t - quasiStaticDecoder dat (Wbar t))
              + D₀ * epsilon ^ 2 := by
  -- Extract constants from the hypotheses
  obtain ⟨D₀, hD₀_pos, hD₀⟩ := hDrift_bound
  obtain ⟨c₀, hc₀_pos, hc₀⟩ := hPD_lower;
  -- Apply the uniform_frob_contraction lemma to obtain the constant lam.
  obtain ⟨lam, hlam_pos, hlam⟩ := uniform_frob_contraction dat Wbar c₀ hc₀_pos (epsilon ^ (2 / L : ℝ)) (by positivity) t_max hc₀;
  refine' ⟨ lam / epsilon ^ ( 2 / L : ℝ ), D₀, _, _, _ ⟩ <;> try positivity;
  intro t ht
  obtain ⟨Vqs_d, hVqs_d⟩ := hVqs_deriv_exists t ht
  have hDelta : HasDerivAt (fun s => V s - quasiStaticDecoder dat (Wbar s)) (-(gradV dat (Wbar t) (V t)) - Vqs_d) t := by
    have := hV_flow_ode t ⟨ ht.1, ht.2.le ⟩;
    rw [ hasDerivAt_pi ] at *;
    exact fun i => by simpa using HasDerivAt.sub ( this i ) ( hVqs_d i ) ;
  have hDelta_deriv : HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat (Wbar s))) ((∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (-(gradV dat (Wbar t) (V t)) - Vqs_d) i j) / matFrobNorm (V t - quasiStaticDecoder dat (Wbar t))) t := by
    convert hasDerivAt_matFrobNorm_of_ne_zero _ _ _ hDelta ( hDelta_nz t ht ) using 1;
  have hDelta_deriv_bound : (∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (-(gradV dat (Wbar t) (V t)) - Vqs_d) i j) ≤ -lam * matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ^ 2 + matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) * matFrobNorm Vqs_d := by
    have hDelta_deriv_bound : (∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (-(gradV dat (Wbar t) (V t))) i j) ≤ -lam * matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ^ 2 := by
      have h_contraction : ∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (gradV dat (Wbar t) (V t)) i j ≥ lam * matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ^ 2 := by
        convert hlam t ( Set.Ico_subset_Icc_self ht ) ( V t - quasiStaticDecoder dat ( Wbar t ) ) using 1;
        · rw [ gradV_eq_delta_mul_A ];
          apply matrix_isUnit_det_of_frob_lower_bound;
          exact mul_pos hc₀_pos ( Real.rpow_pos_of_pos heps ( 2 / L : ℝ ) );
          exact hc₀ t <| Set.Ico_subset_Icc_self ht;
        · unfold matFrobNorm; norm_num [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ;
      norm_num [ Matrix.mulVec, dotProduct ] at * ; linarith;
    have hDelta_deriv_bound : (∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (-Vqs_d) i j) ≤ matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) * matFrobNorm Vqs_d := by
      have hDelta_deriv_bound : |∑ i, ∑ j, (V t - quasiStaticDecoder dat (Wbar t)) i j * (-Vqs_d) i j| ≤ matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) * matFrobNorm Vqs_d := by
        convert cauchy_schwarz_frob ( V t - quasiStaticDecoder dat ( Wbar t ) ) ( -Vqs_d ) using 1 ; norm_num [ matFrobNorm ];
      exact le_of_abs_le hDelta_deriv_bound;
    convert add_le_add ‹∑ i, ∑ j, ( V t - quasiStaticDecoder dat ( Wbar t ) ) i j * ( -gradV dat ( Wbar t ) ( V t ) ) i j ≤ -lam * matFrobNorm ( V t - quasiStaticDecoder dat ( Wbar t ) ) ^ 2› hDelta_deriv_bound using 1 ; simp +decide [ mul_sub ] ; ring;
  refine' ⟨ _, hDelta_deriv, _ ⟩;
  rw [ div_le_iff₀ ];
  · have hVqs_d_bound : matFrobNorm Vqs_d ≤ D₀ * epsilon ^ 2 := by
      convert hD₀ t ht using 1;
      rw [ deriv_pi ];
      · congr! 1;
        ext i j; exact (by
        rw [ deriv_pi ];
        · have := hVqs_d;
          rw [ hasDerivAt_pi ] at this;
          exact HasDerivAt.deriv ( by simpa using HasDerivAt.comp t ( hasDerivAt_pi.1 ( this i ) j ) ( hasDerivAt_id t ) ) ▸ rfl;
        · intro k; exact (by
          have := hVqs_d;
          rw [ hasDerivAt_pi ] at this;
          exact HasDerivAt.differentiableAt ( by simpa using HasDerivAt.comp t ( hasDerivAt_pi.1 ( this i ) k ) ( hasDerivAt_id t ) )));
      · intro i; exact (by
        exact differentiableAt_pi.mp ( hVqs_d.differentiableAt ) i);
    rw [ div_mul_cancel₀ _ ( by positivity ) ] ; nlinarith [ show 0 ≤ matFrobNorm ( V t - quasiStaticDecoder dat ( Wbar t ) ) from Real.sqrt_nonneg _ ] ;
  · unfold matFrobNorm;
    simp +zetaDelta at *;
    contrapose! hDelta_nz;
    exact ⟨ t, ht.1, ht.2, by ext i j; exact sq_eq_zero_iff.mp ( le_antisymm ( le_trans ( Finset.single_le_sum ( fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg ( V t i j - quasiStaticDecoder dat ( Wbar t ) i j ) ) ( Finset.mem_univ i ) |> le_trans ( Finset.single_le_sum ( fun j _ => sq_nonneg ( V t i j - quasiStaticDecoder dat ( Wbar t ) i j ) ) ( Finset.mem_univ j ) ) ) hDelta_nz ) ( sq_nonneg _ ) ) ⟩

/-! ## Section 5.5: Phase A Frozen-Encoder Convergence -/

/-
Triangle inequality for matFrobNorm: ‖A - B‖_F ≤ ‖A‖_F + ‖B‖_F.
-/
lemma matFrobNorm_sub_le {n m : ℕ} (A B : Matrix (Fin n) (Fin m) ℝ) :
    matFrobNorm (A - B) ≤ matFrobNorm A + matFrobNorm B := by
      apply Real.sqrt_le_iff.mpr ⟨ ?_, ?_ ⟩;
      · exact add_nonneg ( Real.sqrt_nonneg _ ) ( Real.sqrt_nonneg _ );
      · unfold matFrobNorm;
        -- By the Cauchy-Schwarz inequality, we have that for any vectors $v$ and $w$ of equal length, $|v \cdot w| \leq \|v\|_2 \|w\|_2$.
        have h_cauchy_schwarz : ∀ (v w : Fin n → Fin m → ℝ), (∑ i, ∑ j, v i j * w i j) ^ 2 ≤ (∑ i, ∑ j, v i j ^ 2) * (∑ i, ∑ j, w i j ^ 2) := by
          intro v w
          have h_cauchy_schwarz : ∀ (u v : Fin n × Fin m → ℝ), (∑ i, u i * v i) ^ 2 ≤ (∑ i, u i ^ 2) * (∑ i, v i ^ 2) := by
            exact?;
          simpa only [ ← Finset.sum_product' ] using h_cauchy_schwarz ( fun p => v p.1 p.2 ) ( fun p => w p.1 p.2 );
        specialize h_cauchy_schwarz ( fun i j => A i j ) ( fun i j => B i j );
        norm_num [ sub_sq ];
        norm_num [ Finset.sum_add_distrib, Finset.mul_sum _ _ _, mul_assoc ];
        norm_num [ ← Finset.mul_sum _ _ _, ← Finset.sum_mul ];
        nlinarith [ show 0 ≤ Real.sqrt ( ∑ i, ∑ j, A i j ^ 2 ) * Real.sqrt ( ∑ i, ∑ j, B i j ^ 2 ) by positivity, Real.mul_self_sqrt ( show 0 ≤ ∑ i, ∑ j, A i j ^ 2 by exact Finset.sum_nonneg fun i hi => Finset.sum_nonneg fun j hj => sq_nonneg _ ), Real.mul_self_sqrt ( show 0 ≤ ∑ i, ∑ j, B i j ^ 2 by exact Finset.sum_nonneg fun i hi => Finset.sum_nonneg fun j hj => sq_nonneg _ ) ]

/-
The Frobenius inner product ⟨ΔV, ΔV · A⟩_F ≥ c₀·ε^{2/L}·‖ΔV‖_F² when
    ‖M·A‖_F ≥ c₀·ε^{2/L}·‖M‖_F for all M. This gives f'(t) ≤ -λ·f(t) with D=0.
-/
lemma frozen_contraction_frob_bound {d : ℕ} (dat : JEPAData d)
    (W₀ : Matrix (Fin d) (Fin d) ℝ)
    (c₀ : ℝ) (hc₀ : 0 < c₀) (epsilon : ℝ) (heps : 0 < epsilon) (L : ℕ) (hL : 2 ≤ L)
    (hPD_lower : ∀ M : Matrix (Fin d) (Fin d) ℝ,
        matFrobNorm (M * (W₀ * dat.SigmaXX * W₀ᵀ)) ≥
          c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M)
    (Delta : Matrix (Fin d) (Fin d) ℝ) :
    ∑ i, ∑ j, Delta i j * (Delta * (W₀ * dat.SigmaXX * W₀ᵀ)) i j ≥
      c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm Delta ^ 2 := by
        -- By Lemma 3, we know that $W₀ * dat.SigmaXX * W₀ᵀ$ is positive definite.
        set A : Matrix (Fin d) (Fin d) ℝ := W₀ * dat.SigmaXX * W₀ᵀ
        have hA_pos : A.PosDef := by
          convert wbarSigma_posDef dat W₀ ( c₀ * epsilon ^ ( 2 / L : ℝ ) ) ( mul_pos hc₀ ( Real.rpow_pos_of_pos heps _ ) ) _ using 1;
          assumption;
        -- From hPD_lower applied to rank-1 matrices of the form (fun i j => if i = k then v j else 0), derive that ∀ v, dotProduct (A.mulVec v) (A.mulVec v) ≥ (c₀ * ε^(2/L))^2 * dotProduct v v.
        have h_rank_one : ∀ v : Fin d → ℝ, dotProduct (A.mulVec v) (A.mulVec v) ≥ (c₀ * epsilon ^ ((2 : ℝ) / L)) ^ 2 * dotProduct v v := by
          intro v;
          -- Let $M$ be the matrix with rows $v$.
          set M : Matrix (Fin d) (Fin d) ℝ := fun i j => v j;
          have := hPD_lower M;
          -- By definition of $M$, we know that $M * A = \sum_{i} v_i A_i$.
          have hMA : matFrobNorm (M * A) ^ 2 = d * dotProduct (A.mulVec v) (A.mulVec v) := by
            unfold matFrobNorm; norm_num [ Matrix.mulVec, dotProduct ] ; ring;
            rw [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ; simp +decide [ M, Matrix.mul_apply, mul_comm ] ; ring;
            have := hA_pos.1; simp_all +decide [ Matrix.IsHermitian, Matrix.mul_apply, mul_comm ] ;
            exact Or.inl ( Finset.sum_congr rfl fun _ _ => by rw [ ← Matrix.ext_iff ] at this; aesop );
          have hMA : matFrobNorm M ^ 2 = d * dotProduct v v := by
            unfold matFrobNorm; norm_num [ Matrix.mulVec, dotProduct ] ; ring;
            rw [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ] ; norm_num [ Finset.mul_sum _ _ _ ] ; ring;
            simp +zetaDelta at *;
            rw [ Finset.mul_sum _ _ _ ];
          rcases d with ( _ | d ) <;> norm_num at *;
          nlinarith [ show 0 ≤ c₀ * epsilon ^ ( 2 / ( L : ℝ ) ) * matFrobNorm M by exact mul_nonneg ( mul_nonneg hc₀.le ( Real.rpow_nonneg heps.le _ ) ) ( Real.sqrt_nonneg _ ), Real.mul_self_sqrt ( show 0 ≤ ( d + 1 : ℝ ) * v ⬝ᵥ v by exact mul_nonneg ( by positivity ) ( Finset.sum_nonneg fun _ _ => mul_self_nonneg _ ) ) ];
        -- Apply pd_quadratic_from_norm_bound to get ∀ v, dotProduct v (A.mulVec v) ≥ c₀ * ε^(2/L) * dotProduct v v.
        have h_quadratic : ∀ v : Fin d → ℝ, dotProduct v (A.mulVec v) ≥ c₀ * epsilon ^ ((2 : ℝ) / L) * dotProduct v v := by
          apply pd_quadratic_from_norm_bound A hA_pos (c₀ * epsilon ^ ((2 : ℝ) / L)) (by positivity) h_rank_one;
        -- Rewrite the sum ∑ i, ∑ j, Delta i j * (Delta * A) i j as ∑ i, dotProduct (Delta i) (A.mulVec (Delta i)) by expanding Matrix.mul_apply.
        have h_sum_expand : ∑ i, ∑ j, Delta i j * (Delta * A) i j = ∑ i, dotProduct (Delta i) (A.mulVec (Delta i)) := by
          simp +decide [ Matrix.mul_apply, dotProduct, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ];
          simp +decide [ Matrix.mulVec, dotProduct, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ];
          exact Finset.sum_congr rfl fun _ _ => Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring );
        rw [ h_sum_expand, matFrobNorm ];
        rw [ Real.sq_sqrt <| Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _ ];
        rw [ Finset.mul_sum _ _ _ ] ; exact Finset.sum_le_sum fun i _ => by simpa [ sq ] using h_quadratic ( Delta i ) ;

/-
Key exponent identity: exp(-(2(L-1)/L) · log(1/ε)) = ε^{2(L-1)/L}.
-/
lemma exp_neg_log_eq_rpow (epsilon : ℝ) (heps : 0 < epsilon) (L : ℕ) (hL : 2 ≤ L) :
    Real.exp (-(2 * ((L : ℝ) - 1) / L) * Real.log (1 / epsilon)) =
      epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
        rw [ Real.rpow_def_of_pos heps, Real.log_div ] <;> norm_num ; ring ; aesop

/-
Exponent monotonicity: ε^{1/L} · ε^{2(L-1)/L} ≤ ε^{2(L-1)/L} for 0 < ε < 1 and L ≥ 2.
    This is because ε^{1/L} ≤ 1.
-/
lemma eps_pow_mul_le (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (L : ℕ) (hL : 2 ≤ L) :
    epsilon ^ ((1 : ℝ) / L) * epsilon ^ (2 * ((L : ℝ) - 1) / L) ≤
      epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
        exact mul_le_of_le_one_left ( Real.rpow_nonneg heps.le _ ) ( Real.rpow_le_one heps.le heps_small.le ( by positivity ) )

/-
ContinuousOn for the frozen-encoder tracking error matFrobNorm.
-/
lemma frozen_tracking_continuousOn {d : ℕ} (dat : JEPAData d)
    (W₀ : Matrix (Fin d) (Fin d) ℝ)
    (V : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (τ_A : ℝ) (hτ_A : 0 < τ_A)
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 τ_A,
        HasDerivAt V (-(gradV dat W₀ (V t))) t) :
    ContinuousOn (fun t => matFrobNorm (V t - quasiStaticDecoder dat W₀)) (Set.Icc 0 τ_A) := by
  -- Since $V$ is differentiable on $[0, \tau_A]$, it is continuous on this interval.
  have hV_cont : ContinuousOn V (Set.Icc 0 τ_A) := by
    intro t ht;
    have := hV_flow_ode t ht;
    rw [ hasDerivAt_pi ] at this;
    exact tendsto_pi_nhds.mpr fun i => ( this i |> HasDerivAt.continuousAt |> ContinuousAt.continuousWithinAt );
  refine' ContinuousOn.sqrt _;
  fun_prop

/-- **Lemma (Frozen-encoder Phase A convergence).**
    When W̄ is held fixed at W₀ and V evolves under the decoder gradient flow
    V̇(t) = -gradV dat W₀ (V t), the tracking error f(t) = ‖V(t) - V_qs(W₀)‖_F
    decays exponentially. Starting from ‖V(0)‖_F ≤ K₀·ε^{1/L} and ‖V_qs(W₀)‖_F ≤ K_qs·ε^{1/L}
    (from `hK_qs`), with the Frobenius PD lower bound ‖M·(W₀ Σˣˣ W₀ᵀ)‖_F ≥ c₀·ε^{2/L}·‖M‖_F,
    after the logarithmic Phase A time

        τ_A = (2(L-1)/L) / c₀ · ε^{-2/L} · log(1/ε)

    the tracking error satisfies
        f(τ_A) ≤ (K₀ + K_qs) · ε^{2(L-1)/L}.

    The constant K₀ + K_qs is ε-independent (both bounds from problem data); this is the
    genuine reformulation replacing the previous vacuous existential witness.

    This lemma discharges hypothesis (R1) `hPhaseA` of `JEPA_rho_ordering`.

    PROVIDED SOLUTION

    Let f(t) = matFrobNorm(V(t) - quasiStaticDecoder dat W₀).
    Let ΔV(t) = V(t) - quasiStaticDecoder dat W₀ (the quasi-static decoder is constant
    since W₀ is fixed throughout Phase A).

    Step 1: Compute the ODE for ΔV(t). Since d/dt[quasiStaticDecoder dat W₀] = 0:
            ΔV̇(t) = V̇(t) = -gradV dat W₀ (V t).
            By the identity gradV dat W₀ V = (V - quasiStaticDecoder dat W₀) * (W₀ * Σˣˣ * W₀ᵀ)
            (this is the linearisation around the quasi-static decoder; use gradV_eq_delta_mul_A
            from Basic.lean or unfold gradV directly):
            ΔV̇(t) = -ΔV(t) * (W₀ * dat.SigmaXX * W₀ᵀ).
            Let A := W₀ * dat.SigmaXX * W₀ᵀ (constant positive-semidefinite matrix).

    Step 2: Derive the scalar ODE for f(t) when ΔV(t) ≠ 0.
            Extract the c₀ and the uniform lower bound from hPD_lower:
            obtain ⟨c₀, hc₀, hPD⟩ := hPD_lower.
            Set lam := c₀ * epsilon ^ ((2 : ℝ) / L). Then lam > 0.
            On the set where ΔV(t) ≠ 0, apply hasDerivAt_matFrobNorm_of_ne_zero
            (using hDelta_nz for t ∈ Set.Ico 0 τ_A):
            f'(t) = ⟨ΔV(t), ΔV̇(t)⟩_F / f(t)
                  = -⟨ΔV(t), ΔV(t) * A⟩_F / f(t).
            By hPD applied to M = ΔV(t): ⟨ΔV, ΔV * A⟩_F ≥ lam * f(t)^2.
            Dividing by f(t) > 0: f'(t) ≤ -lam * f(t).
            This is a pure contraction (drift D = 0).

    Step 3: Apply contractive_gronwall_decay (Lemmas.lean, Section 4) with D = 0.
            Hypotheses:
            - hT := hτ_A (τ_A > 0)
            - hlam := lam > 0
            - hD := le_refl 0
            - hf_cont: f is continuous on [0, τ_A] from hV_flow_ode + matrix operations
            - hf_nn: f(t) ≥ 0 by Real.sqrt_nonneg
            - hf_deriv: for t ∈ Set.Ico 0 τ_A, f'(t) ≤ -lam * f(t) + 0 (from Step 2)
            Conclusion: ∀ t ∈ [0, τ_A], f(t) ≤ f(0) * Real.exp(-lam * t).

    Step 4: Bound f(0) using triangle inequality.
            f(0) = matFrobNorm(V 0 - quasiStaticDecoder dat W₀)
                 ≤ matFrobNorm(V 0) + matFrobNorm(quasiStaticDecoder dat W₀).
            From hV_init: obtain ⟨K₀, hK₀, hV₀⟩. So matFrobNorm(V 0) ≤ K₀ * ε^{1/L}.
            From hK_qs: obtain ⟨K_qs, hK_qs_pos, hVqs₀⟩. So matFrobNorm(V_qs(W₀)) ≤ K_qs * ε^{1/L}.
            Therefore f(0) ≤ (K₀ + K_qs) * epsilon ^ ((1 : ℝ) / L).

    Step 5: Evaluate at t = τ_A using hτ_A_def to connect τ_A with ε.
            Extract ⟨c₀', hc₀', hτ⟩ := hτ_A_def. Use c₀ = c₀' (both from hPD_lower, same bound).
            lam * τ_A = c₀ * ε^{2/L} * [(2(L-1)/L) / c₀ * ε^{-2/L} * log(1/ε)]
                      = (2(L-1)/L) * log(1/ε).
            Real.exp(-lam * τ_A) = Real.exp(-(2(L-1)/L) * log(1/ε))
                                  = (1/ε)^{-(2(L-1)/L)}
                                  = ε^{2(L-1)/L}.
            Rewrite using Real.exp_log (heps > 0) and Real.rpow_natCast or Real.exp_mul_log.
            Key: Real.exp (-(2 * ((L:ℝ) - 1) / L) * Real.log (1 / epsilon)) = epsilon ^ (2 * ((L:ℝ) - 1) / L).
            Proof: exp(a * log(1/ε)) = (1/ε)^a = ε^{-a}, so exp(-a * log(1/ε)) = ε^a.
            Use Real.rpow_def_of_pos heps and Real.log_inv.

    Step 6: Combine Steps 3–5.
            f(τ_A) ≤ f(0) * Real.exp(-lam * τ_A)
                   ≤ (K₀ + K_qs) * ε^{1/L} * ε^{2(L-1)/L}
                   = (K₀ + K_qs) * ε^{(2L-1)/L}.
            Since (2L-1)/L ≥ 2(L-1)/L for L ≥ 2 (both equal 2 - 1/L vs 2 - 2/L; since 1/L ≤ 2/L),
            we have ε^{(2L-1)/L} ≤ ε^{2(L-1)/L} (because ε < 1).
            So f(τ_A) ≤ (K₀ + K_qs) * ε^{2(L-1)/L}.
            Witness C_A := K₀ + K_qs.
            Use Real.rpow_le_rpow_of_exponent_le (heps_small.le) and exponent comparison.

    Step 7: Handle the zero case separately.
            If f(0) = 0 then ΔV(0) = 0 (V(0) = V_qs(W₀)) and by uniqueness of ODE solutions
            ΔV(t) = 0 for all t, so f(τ_A) = 0 ≤ (K₀ + K_qs) * ε^{2(L-1)/L}.
            In practice: if matFrobNorm(V 0 - quasiStaticDecoder dat W₀) = 0, then
            contractive_gronwall_decay gives f(τ_A) ≤ 0 * exp(...) + 0 = 0. -/
lemma frozen_encoder_convergence {d : ℕ} (hd : 0 < d) (dat : JEPAData d)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    -- Fixed encoder W₀ (Phase A: frozen)
    (W₀ : Matrix (Fin d) (Fin d) ℝ)
    -- Explicit ε-independent constants (K₀ + K_qs is the final bound constant)
    (K₀ K_qs : ℝ) (hK₀ : 0 < K₀) (hK_qs_pos : 0 < K_qs)
    -- Initial bound on V (‖V(0)‖_F ≤ K₀ · ε^{1/L})
    (V : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (hV_init : matFrobNorm (V 0) ≤ K₀ * epsilon ^ ((1 : ℝ) / L))
    -- Quasi-static decoder norm bound (‖V_qs(W₀)‖_F ≤ K_qs · ε^{1/L}); the quasi-static decoder
    -- V_qs(W₀) = W₀ Σʸˣ W₀ᵀ (W₀ Σˣˣ W₀ᵀ)⁻¹ scales like W₀ (order ε^{1/L}), so K_qs depends
    -- only on the spectral norms of Σˣˣ and Σʸˣ, not on ε.
    (hK_qs : matFrobNorm (quasiStaticDecoder dat W₀) ≤ K_qs * epsilon ^ ((1 : ℝ) / L))
    -- V satisfies the frozen-encoder gradient flow on [0, τ_A]
    (τ_A : ℝ) (hτ_A : 0 < τ_A)
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 τ_A,
        HasDerivAt V (-(gradV dat W₀ (V t))) t)
    -- Frobenius PD lower bound: ‖M · (W₀ Σˣˣ W₀ᵀ)‖_F ≥ c₀ · ε^{2/L} · ‖M‖_F
    (c₀ : ℝ) (hc₀ : 0 < c₀)
    (hPD_lower : ∀ M : Matrix (Fin d) (Fin d) ℝ,
        matFrobNorm (M * (W₀ * dat.SigmaXX * W₀ᵀ)) ≥
          c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M)
    -- τ_A is the logarithmic Phase A timescale: τ_A = (2(L-1)/L) / c₀ · ε^{-2/L} · log(1/ε)
    (hτ_A_def : τ_A = (2 * ((L : ℝ) - 1) / L) / c₀ * epsilon ^ (-(2 : ℝ) / L) *
                      Real.log (1 / epsilon))
    -- Tracking error is nonzero on (0, τ_A) (or zero, trivially satisfied)
    (hDelta_nz : ∀ t ∈ Set.Ico 0 τ_A,
        V t - quasiStaticDecoder dat W₀ ≠ 0)
    : matFrobNorm (V τ_A - quasiStaticDecoder dat W₀) ≤
        (K₀ + K_qs) * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
  -- Apply contractive_gronwall_decay with D=0 and λ = c₀ * epsilon^(2/L) to obtain the inequality.
  have h_gronwall : matFrobNorm (V τ_A - quasiStaticDecoder dat W₀) ≤ (K₀ + K_qs) * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
    have h_deriv_bound : ∀ t ∈ Set.Ico 0 τ_A, ∃ f' : ℝ, HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat W₀)) f' t ∧ f' ≤ -c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm (V t - quasiStaticDecoder dat W₀) := by
      intro t ht
      obtain ⟨f', hf'_deriv, hf'_bound⟩ : ∃ f' : ℝ, HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat W₀)) f' t ∧ f' = (∑ i, ∑ j, (V t - quasiStaticDecoder dat W₀) i j * (-(gradV dat W₀ (V t))) i j) / matFrobNorm (V t - quasiStaticDecoder dat W₀) := by
        have h_deriv : HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat W₀)) ((∑ i, ∑ j, (V t - quasiStaticDecoder dat W₀) i j * (-gradV dat W₀ (V t)) i j) / matFrobNorm (V t - quasiStaticDecoder dat W₀)) t := by
          have h_deriv : HasDerivAt (fun s => V s - quasiStaticDecoder dat W₀) (-gradV dat W₀ (V t)) t := by
            have := hV_flow_ode t ⟨ ht.1, ht.2.le ⟩;
            rw [ hasDerivAt_pi ] at *;
            exact fun i => by simpa using this i |> HasDerivAt.sub <| hasDerivAt_const _ _;
          convert hasDerivAt_matFrobNorm_of_ne_zero _ _ _ h_deriv _ using 1 ; aesop;
        exact ⟨ _, h_deriv, rfl ⟩;
      refine' ⟨ f', hf'_deriv, _ ⟩;
      rw [ hf'_bound, div_le_iff₀ ];
      · have := frozen_contraction_frob_bound dat W₀ c₀ hc₀ epsilon heps L hL hPD_lower ( V t - quasiStaticDecoder dat W₀ );
        rw [ gradV_eq_delta_mul_A ] at *;
        · norm_num [ Matrix.mul_apply ] at * ; linarith;
        · apply matrix_isUnit_det_of_frob_lower_bound;
          exact mul_pos hc₀ ( Real.rpow_pos_of_pos heps ( 2 / L ) );
          exact hPD_lower;
        · apply matrix_isUnit_det_of_frob_lower_bound;
          exact mul_pos hc₀ ( Real.rpow_pos_of_pos heps ( 2 / L ) );
          exact hPD_lower;
      · refine' Real.sqrt_pos.mpr _;
        contrapose! hDelta_nz;
        exact ⟨ t, ht, by ext i j; exact sq_eq_zero_iff.mp ( le_antisymm ( le_trans ( Finset.single_le_sum ( fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg ( ( V t - quasiStaticDecoder dat W₀ ) i j ) ) ( Finset.mem_univ i ) |> le_trans ( Finset.single_le_sum ( fun j _ => sq_nonneg ( ( V t - quasiStaticDecoder dat W₀ ) i j ) ) ( Finset.mem_univ j ) ) ) hDelta_nz ) ( sq_nonneg _ ) ) ⟩
    -- Apply the contractive_gronwall_decay lemma with D=0 to get the inequality.
    have h_gronwall : matFrobNorm (V τ_A - quasiStaticDecoder dat W₀) ≤ matFrobNorm (V 0 - quasiStaticDecoder dat W₀) * Real.exp (-c₀ * epsilon ^ ((2 : ℝ) / L) * τ_A) := by
      have h_gronwall : ∀ t ∈ Set.Icc 0 τ_A, matFrobNorm (V t - quasiStaticDecoder dat W₀) ≤ matFrobNorm (V 0 - quasiStaticDecoder dat W₀) * Real.exp (-c₀ * epsilon ^ ((2 : ℝ) / L) * t) := by
        have := @contractive_gronwall_decay;
        convert @this τ_A hτ_A ( fun t => matFrobNorm ( V t - quasiStaticDecoder dat W₀ ) ) ( c₀ * epsilon ^ ( 2 / ( L : ℝ ) ) ) 0 ( mul_pos hc₀ ( Real.rpow_pos_of_pos heps _ ) ) le_rfl ( ?_ ) ( ?_ ) ( ?_ ) using 1;
        · norm_num;
        · apply_rules [ frozen_tracking_continuousOn ];
        · exact fun t ht => Real.sqrt_nonneg _;
        · exact fun t ht => by obtain ⟨ f', hf₁, hf₂ ⟩ := h_deriv_bound t ht; exact ⟨ f', hf₁, by linarith ⟩ ;
      exact h_gronwall τ_A ⟨ hτ_A.le, le_rfl ⟩;
    -- Substitute the bound for matFrobNorm (V 0 - quasiStaticDecoder dat W₀) into the inequality from h_gronwall.
    have h_subst : matFrobNorm (V τ_A - quasiStaticDecoder dat W₀) ≤ (K₀ + K_qs) * epsilon ^ ((1 : ℝ) / L) * Real.exp (-c₀ * epsilon ^ ((2 : ℝ) / L) * τ_A) := by
      refine le_trans h_gronwall ?_;
      gcongr;
      exact le_trans ( matFrobNorm_sub_le _ _ ) ( by linarith );
    -- Simplify the exponent in the inequality.
    have h_exp_simplified : Real.exp (-c₀ * epsilon ^ ((2 : ℝ) / L) * τ_A) = epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
      rw [ hτ_A_def ] ; ring;
      norm_num [ Real.rpow_def_of_pos heps, mul_assoc, mul_comm c₀, hc₀.ne', show L ≠ 0 by positivity ] ; ring;
      norm_num [ mul_assoc, ← Real.exp_add ] ; ring;
    refine le_trans h_subst ?_;
    rw [ h_exp_simplified, mul_assoc ];
    exact mul_le_mul_of_nonneg_left ( mul_le_of_le_one_left ( by positivity ) ( Real.rpow_le_one ( by positivity ) heps_small.le ( by positivity ) ) ) ( by positivity );
  exact h_gronwall

