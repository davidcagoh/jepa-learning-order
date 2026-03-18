import Mathlib

/-!
# JEPA Learns Influential Features First
## A Proof Without Simultaneous Diagonalizability

David Goh — March 2026

We formalise the result that a depth-L ≥ 2 linear JEPA model, trained from
small random initialisation, learns features in decreasing order of their
generalised regression coefficient ρ*, even when the input and cross-covariance
matrices share no common eigenbasis.
-/

variable {d : ℕ} (hd : 0 < d)

/-! ## Section 1 & 2: Definitions -/

/-- The input covariance matrix Σˣˣ = E[xxᵀ], required to be positive definite. -/
structure JEPAData (d : ℕ) where
  /-- Input covariance Σˣˣ ∈ ℝ^{d×d}, positive definite -/
  SigmaXX : Matrix (Fin d) (Fin d) ℝ
  /-- Cross-covariance Σʸˣ = E[yxᵀ] ∈ ℝ^{d×d} -/
  SigmaYX : Matrix (Fin d) (Fin d) ℝ
  /-- Output covariance Σʸʸ = E[yyᵀ] ∈ ℝ^{d×d} -/
  SigmaYY : Matrix (Fin d) (Fin d) ℝ
  /-- Σˣˣ is positive definite -/
  hSigmaXX_pos : Matrix.PosDef SigmaXX

/-- Definition 2.1. The regression operator ℛ = (Σˣˣ)⁻¹ Σʸˣ. -/
def regressionOperator (dat : JEPAData d) : Matrix (Fin d) (Fin d) ℝ :=
  dat.SigmaXX⁻¹ * dat.SigmaYX

/-- The JEPA loss function.
    ℒ(W̄, V) = ½ tr(V W̄ Σˣˣ W̄ᵀ Vᵀ) - tr(V W̄ Σʸˣ) + ½ tr(W̄ Σʸʸ W̄ᵀ) -/
noncomputable def JEPALoss (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  (1 / 2) * Matrix.trace (V * Wbar * dat.SigmaXX * Wbarᵀ * Vᵀ)
  - Matrix.trace (V * Wbar * dat.SigmaYX)
  + (1 / 2) * Matrix.trace (Wbar * dat.SigmaYY * Wbarᵀ)

/-- The gradient of the JEPA loss with respect to V:
    ∇_V ℒ = V W̄ Σˣˣ W̄ᵀ - W̄ Σʸˣ W̄ᵀ  -- TODO: check sign convention matches paper -/
noncomputable def gradV (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  V * Wbar * dat.SigmaXX * Wbarᵀ - Wbar * dat.SigmaYX

/-- The gradient of the JEPA loss with respect to W̄:
    ∇_{W̄} ℒ = Vᵀ (V W̄ Σˣˣ - W̄ Σʸˣ) -/
noncomputable def gradWbar (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Vᵀ * (V * Wbar * dat.SigmaXX - Wbar * dat.SigmaYX)  -- TODO: check transpose conventions

/-- Definition 2.2. A generalised eigenpair (v, ρ) satisfies Σʸˣ v = ρ Σˣˣ v
    with the Σˣˣ-orthonormality condition vᵀ Σˣˣ v = μ > 0. -/
structure GenEigenpair (dat : JEPAData d) where
  /-- The generalised eigenvector v* ∈ ℝ^d -/
  v : Fin d → ℝ
  /-- The generalised eigenvalue ρ* > 0 -/
  rho : ℝ
  /-- The Σˣˣ-norm squared μ = vᵀ Σˣˣ v > 0 -/
  mu : ℝ
  /-- Generalised eigenvalue equation: Σʸˣ v = ρ Σˣˣ v -/
  heig : dat.SigmaYX.mulVec v = rho • dat.SigmaXX.mulVec v
  /-- Positivity of ρ -/
  hrho_pos : 0 < rho
  /-- Positivity of μ = vᵀ Σˣˣ v -/
  hmu_pos : 0 < mu
  /-- μ = vᵀ Σˣˣ v -/
  hmu_def : mu = Matrix.dotProduct v (dat.SigmaXX.mulVec v)

/-- The complete generalised eigenbasis: d eigenpairs with strictly decreasing eigenvalues. -/
structure GenEigenbasis (dat : JEPAData d) where
  /-- The r-th generalised eigenpair -/
  pairs : Fin d → GenEigenpair dat
  /-- Eigenvalues are strictly decreasing: ρ₁* > ρ₂* > … > ρ_d* -/
  hstrictly_decreasing : ∀ r s : Fin d, r < s → (pairs s).rho < (pairs r).rho
  /-- All eigenvalues positive (already in GenEigenpair, but stated globally) -/
  hpos : ∀ r : Fin d, 0 < (pairs r).rho
  /-- Σˣˣ-biorthogonality: v_rᵀ Σˣˣ v_s = δ_{rs} μ_r -/
  hbiorthog : ∀ r s : Fin d, r ≠ s →
    Matrix.dotProduct (pairs r).v (dat.SigmaXX.mulVec (pairs s).v) = 0

/-- The dual left basis u* satisfying u_rᵀ Σˣˣ v_s = δ_{rs} μ_r.
    Here we define u_r as the left generalised eigenvector. -/
noncomputable def dualBasis (dat : JEPAData d) (eb : GenEigenbasis dat) :
    Fin d → (Fin d → ℝ) :=
  fun r => dat.SigmaXX.mulVec (eb.pairs r).v  -- TODO: check: this gives Σˣˣ v_r, dual under ⟨·, Σˣˣ ·⟩

/-- The projected covariance λ_r* = ρ_r* · μ_r. -/
noncomputable def projectedCovariance (dat : JEPAData d) (eb : GenEigenbasis dat)
    (r : Fin d) : ℝ :=
  (eb.pairs r).rho * (eb.pairs r).mu

/-- Definition 2.3. The diagonal amplitude σ_r(t) = u_rᵀ W̄(t) v_r*. -/
noncomputable def diagAmplitude (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) (r : Fin d) : ℝ :=
  Matrix.dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs r).v)

/-- Definition 2.3. The off-diagonal amplitude c_{rs}(t) = u_rᵀ W̄(t) v_s* for r ≠ s. -/
noncomputable def offDiagAmplitude (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) (r s : Fin d) : ℝ :=
  Matrix.dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs s).v)

/-- The balanced network preconditioning coefficient P_{rs}(t) for depth L.
    P_{rs} = Σ_{a=1}^{L} σ_r^{2(L-a)/L} · σ_s^{2(a-1)/L} -/
noncomputable def preconditioner (L : ℕ) (sigma_r sigma_s : ℝ) : ℝ :=
  ∑ a : Fin L,
    sigma_r ^ (2 * (L - (a.val + 1)) / L)  -- TODO: check: real-valued exponents need rpow
    * sigma_s ^ (2 * a.val / L)

/-! ## Section 3: Key Lemma — Gradient Decouples in the Generalised Eigenbasis -/

/-- **Lemma 3.1 (Gradient projection).** For any W̄ and V,
    (-∇_{W̄} ℒ) v_r* = Vᵀ (ρ_r* I - V) W̄ Σˣˣ v_r*.

    PROVIDED SOLUTION
    Step 1: Expand -∇_{W̄} ℒ = Vᵀ W̄ Σʸˣ - Vᵀ V W̄ Σˣˣ.
    Step 2: Apply to v_r* and substitute the generalised eigenvalue equation
            Σʸˣ v_r* = ρ_r* Σˣˣ v_r* (from GenEigenpair.heig).
    Step 3: Factor out Vᵀ to obtain Vᵀ (ρ_r* W̄ Σˣˣ v_r* - V W̄ Σˣˣ v_r*)
            = Vᵀ (ρ_r* I - V) W̄ Σˣˣ v_r*. -/
lemma gradient_projection (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) (r : Fin d) :
    (-(gradWbar dat Wbar V)).mulVec (eb.pairs r).v =
    Vᵀ.mulVec ((eb.pairs r).rho • V.mulVec (Wbar.mulVec (dat.SigmaXX.mulVec (eb.pairs r).v))
              - V.mulVec (Wbar.mulVec (dat.SigmaXX.mulVec (eb.pairs r).v))) := by
  sorry

/-! ## Section 4: Initialisation and the Balanced Network -/

/-- **Assumption 4.1 (Balanced initialisation).**
    Each layer starts at W^a(0) = ε^{1/L} U^a with U^a orthogonal.
    The decoder starts at V(0) = ε^{1/L} U^v with U^v orthogonal.
    Balancedness: W^{a+1}(t)ᵀ W^{a+1}(t) = W^a(t) W^a(t)ᵀ for all t. -/
structure BalancedInit (d L : ℕ) (epsilon : ℝ) where
  /-- The L encoder layers at time 0 -/
  W0 : Fin L → Matrix (Fin d) (Fin d) ℝ
  /-- The decoder at time 0 -/
  V0 : Matrix (Fin d) (Fin d) ℝ
  /-- Each encoder layer is ε^{1/L} times an orthogonal matrix -/
  hW_orth : ∀ a : Fin L, Matrix.IsOrthogonal (epsilon ^ (-(1 : ℝ) / L) • W0 a)  -- TODO: check Matrix.IsOrthogonal name
  /-- Decoder is ε^{1/L} times an orthogonal matrix -/
  hV_orth : Matrix.IsOrthogonal (epsilon ^ (-(1 : ℝ) / L) • V0)  -- TODO: check
  /-- Balancedness condition: W^{a+1}(0)ᵀ W^{a+1}(0) = W^a(0) W^a(0)ᵀ -/
  hbalanced : ∀ a : Fin (L - 1),
    (W0 ⟨a.val + 1, by omega⟩)ᵀ * W0 ⟨a.val + 1, by omega⟩ =
    W0 ⟨a.val, by omega⟩ * (W0 ⟨a.val, by omega⟩)ᵀ
  /-- Positivity of scale -/
  heps_pos : 0 < epsilon

/-! ## Section 5: Timescale Separation and the Quasi-Static Decoder -/

/-- **Definition 5.1 (Quasi-static fixed point).**
    For fixed W̄, the minimiser of ℒ over V is
    V_qs(W̄) = W̄ Σʸˣ W̄ᵀ (W̄ Σˣˣ W̄ᵀ)⁻¹.
    Obtained by setting ∇_V ℒ = 0 and solving. -/
noncomputable def quasiStaticDecoder (dat : JEPAData d)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Wbar * dat.SigmaYX * Wbarᵀ * (Wbar * dat.SigmaXX * Wbarᵀ)⁻¹

/-- **Lemma 5.2 (Quasi-static decoder approximation).**
    For L ≥ 2 and initialisation scale ε ≪ 1, the decoder satisfies
    ‖V(t) - V_qs(W̄(t))‖ = O(ε^{2(L-1)/L}) uniformly for t ∈ [0, t_max*].

    PROVIDED SOLUTION
    Two-phase argument:

    Phase A (t ∈ [0, τ_A], τ_A = O(ε^{-2/L})):
    Step 1: Show encoder moves at rate ‖Ẇ̄‖ = O(ε²) during Phase A.
    Step 2: Over duration τ_A, encoder displacement is O(ε^{(2L-2)/L}) — negligible.
    Step 3: With W̄ ≈ ε^{1/L} I fixed, the decoder ODE ̇V = -ε^{2/L}(V Σˣˣ - Σʸˣ)
            is exactly solvable: V(t) = Σʸˣ(Σˣˣ)⁻¹(I - exp(-ε^{2/L} Σˣˣ t)) + ε^{1/L} exp(-ε^{2/L} Σˣˣ t).
    Step 4: Exponential convergence to V_qs(ε^{1/L} I) = Σʸˣ(Σˣˣ)⁻¹ on timescale O(ε^{-2/L}).
    Step 5: Encoder changes in eigenbasis during Phase A:
            Δσ_r^A = O(ρ_r* ε^{(2L-3)/L}), Δc_{rs}^A = O(ε^{(2L-3)/L}).
            Both are o(ε^{1/L}) for L ≥ 2.

    Phase B (t ∈ [τ_A, t_max*]):
    Step 6: Define ΔV(t) = V(t) - V_qs(W̄(t)).
            Using ∇_V ℒ = (V - V_qs) W̄ Σˣˣ W̄ᵀ, the deviation satisfies:
            ΔV̇ = -ΔV · W̄ Σˣˣ W̄ᵀ - d/dt V_qs(W̄).
    Step 7: Contraction rate α(t) = λ_min(W̄ Σˣˣ W̄ᵀ) = O(ε^{2/L}).
    Step 8: Drift rate ‖∂_{W̄} V_qs‖_op · ‖Ẇ̄‖ = O(ε²).
    Step 9: By variation of constants: ‖ΔV(t)‖_ss ≲ O(ε²)/O(ε^{2/L}) = O(ε^{2(L-1)/L}).
    Step 10: For L ≥ 2, this → 0 as ε → 0. -/
lemma quasiStatic_approx (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    -- V and Wbar as functions of time
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    -- Gradient flow equations (hypotheses encoding the ODE)
    (hV_flow : ∀ t : ℝ, deriv (fun t => ‖V t - quasiStaticDecoder dat (Wbar t)‖) t ≤ 0)  -- TODO: strengthen to exact ODE
    -- Bound: ‖V(t) - V_qs(W̄(t))‖ = O(ε^{2(L-1)/L})
    : ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      ‖V t - quasiStaticDecoder dat (Wbar t)‖ ≤ C * epsilon ^ (2 * (L - 1) / L) := by
  sorry

/-! ## Section 6: Diagonal Dynamics — The Littwin ODE -/

/-- **Proposition 6.1 (Effective diagonal ODE).**
    Under approximate alignment c_{rs} = O(ε^{1/L}), the diagonal amplitude σ_r(t) satisfies
    σ̇_r = σ_r^{3-1/L} λ_r* - (σ_r³ λ_r*) / ρ_r*,   λ_r* = ρ_r* μ_r,
    to leading order in ε.

    PROVIDED SOLUTION
    Step 1: Project the encoder gradient (Lemma 3.1) onto mode r.
            Writing v_r = u_rᵀ V u_r* for the r-th decoder gain:
            u_rᵀ (-∇_{W̄} ℒ) v_r* = v_r(ρ_r* - v_r) σ_r μ_r.
    Step 2: Project the decoder gradient onto mode r:
            v̇_r = -σ_r² μ_r (v_r - ρ_r*).
    Step 3: This 2D system (σ_r, v_r) is identical in structure to the diagonal
            JEPA system of Littwin et al. (2024), Theorem 4.2,
            with parameters (λ_r*, μ_r, ρ_r*) in place of (λ_i, σ_i², ρ_i).
    Step 4: Cross-mode coupling via c_{rs} enters only at next order O(ε^{1/L}).
    Step 5: Apply the Littwin et al. balanced-network conservation law to reduce
            the 2D system to the stated scalar ODE. -/
lemma diagonal_ODE (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r : Fin d)
    (sigma_r : ℝ → ℝ)  -- diagonal amplitude as function of time
    (v_r : ℝ → ℝ)      -- r-th decoder gain as function of time
    -- Hypothesis: off-diagonal amplitudes are O(ε^{1/L})
    (hoff_diag_small : ∀ s : Fin d, s ≠ r → ∃ C : ℝ, ∀ t : ℝ,
      |v_r t| ≤ C * epsilon ^ ((1 : ℝ) / L))  -- TODO: this hypothesis needs refinement
    -- Leading-order ODE for σ_r:
    -- σ̇_r = σ_r^{3-1/L} λ_r* - σ_r³ λ_r* / ρ_r*
    : ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t →
      |deriv sigma_r t -
        (sigma_r t ^ (3 - (1 : ℝ) / L) * projectedCovariance dat eb r
         - sigma_r t ^ 3 * projectedCovariance dat eb r / (eb.pairs r).rho)|
      ≤ C * epsilon ^ ((1 : ℝ) / L) := by
  sorry

/-- **Corollary 6.2 (Critical time formula).**
    The critical time t̃_r* at which σ_r reaches fraction p of its asymptote
    σ_r* = (ρ_r*)^{1/2} μ_r^{1/2} is
    t̃_r* = (1/λ_r*) Σ_{n=1}^{2L-1} L / (n ρ_r*^{2L-n-1} ε^{n/L}) + Θ(log ε).
    Leading order: t̃_r* ≈ L / (λ_r* ρ_r*^{2L-2} ε^{1/L}).

    Since t̃_r* is strictly decreasing in ρ_r*, features with higher ρ* reach
    their asymptote first (for ε sufficiently small and off-diagonal corrections
    remaining O(ε^{1/L})).

    PROVIDED SOLUTION
    Step 1: Solve the scalar ODE from Proposition 6.1 for σ_r(t).
    Step 2: Invert to get the time t at which σ_r = p · σ_r*.
    Step 3: Expand the resulting expression in powers of ε^{1/L}, identifying
            the coefficients L / (n ρ_r*^{2L-n-1}) for n = 1, …, 2L-1.
    Step 4: Show ∂(t̃_r*)/∂(ρ_r*) < 0 by differentiating the leading term
            L / (λ_r* ρ_r*^{2L-2} ε^{1/L}) with respect to ρ_r*,
            using λ_r* = ρ_r* μ_r and noting (2L-3) > 0 for L ≥ 2. -/
lemma critical_time_formula (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r : Fin d)
    (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    -- The asymptotic amplitude is σ_r* = sqrt(ρ_r* · μ_r)
    let sigma_r_star := Real.sqrt ((eb.pairs r).rho * (eb.pairs r).mu)
    -- The leading-order critical time
    let t_crit_leading := (L : ℝ) /
      (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
    -- There exist constants C₁, C₂ such that t̃_r* lies between the bounds
    ∃ C₁ C₂ : ℝ, t_crit_leading - C₁ * |Real.log epsilon| ≤ C₂ ∧
      C₂ ≤ t_crit_leading + C₁ * |Real.log epsilon| := by
  sorry

/-- **Corollary 6.2 (Ordering).** Higher ρ* implies smaller critical time.
    For ρ_r* > ρ_s*, we have t̃_r* < t̃_s* for all sufficiently small ε.

    PROVIDED SOLUTION
    Step 1: Use the leading-order formula from critical_time_formula.
            t̃_r* ≈ L / (λ_r* ρ_r*^{2L-2} ε^{1/L}).
    Step 2: Compare: t̃_r* / t̃_s* ≈ (λ_s* ρ_s*^{2L-2}) / (λ_r* ρ_r*^{2L-2}).
    Step 3: Since ρ_r* > ρ_s* > 0 and L ≥ 2, we have ρ_r*^{2L-1} > ρ_s*^{2L-1}.
    Step 4: The leading-order gap t̃_s* - t̃_r* is O(ε^{-1/L}), which dominates
            the subleading corrections of size O(ε^{2(L-1)/L} |log ε|). -/
lemma critical_time_ordering (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (r s : Fin d) (hrs : (eb.pairs s).rho < (eb.pairs r).rho) :
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
    -- t̃_r* < t̃_s*: the leading-order critical time for r is strictly less than for s
    (L : ℝ) / (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
    < (L : ℝ) / (projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L)) := by
  sorry

/-! ## Section 7: Off-Diagonal Dynamics and the Grönwall Bound -/

/-- **Lemma 7.1 (Off-diagonal ODE).**
    Under Lemma 5.2, for r ≠ s:
    ċ_{rs} = -P_{rs}(t) · ρ_r*(ρ_r* - ρ_s*) μ_s · c_{rs} + O(ε^{(2L-1)/L}).

    PROVIDED SOLUTION
    Step 1: Project Lemma 3.1 onto the (r,s) off-diagonal entry.
            With V ≈ diag(ρ_r*):
            u_rᵀ (-∇_{W̄} ℒ) v_s* = u_rᵀ Vᵀ (ρ_s* I - V) W̄ Σˣˣ v_s*
                                    = ρ_r* (ρ_s* - ρ_r*) μ_s c_{rs}.
    Step 2: Apply the balanced-network preconditioning factor P_{rs}
            (Arora et al. 2019) to get the time derivative of c_{rs}.
    Step 3: The O(ε^{2(L-1)/L}) error in V (Lemma 5.2) induces a forcing term
            of size O(ε^{2(L-1)/L}) · ‖W̄‖ = O(ε^{2(L-1)/L}) · O(ε^{1/L}) = O(ε^{(2L-1)/L}). -/
lemma offDiag_ODE (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r s : Fin d) (hrs : r ≠ s)
    (c_rs : ℝ → ℝ)    -- off-diagonal amplitude as function of time
    (sigma_r sigma_s : ℝ → ℝ)  -- diagonal amplitudes
    (t_max : ℝ) (ht_max : 0 < t_max) :
    ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      |deriv c_rs t
        + preconditioner L (sigma_r t) (sigma_s t)
          * (eb.pairs r).rho * ((eb.pairs r).rho - (eb.pairs s).rho) * (eb.pairs s).mu
          * c_rs t|
      ≤ C * epsilon ^ ((2 * L - 1 : ℝ) / L) := by
  sorry

/-- **Lemma 7.2 (Integral bound — the heart of the depth condition).**
    For L ≥ 2 and all r, s:
    ∫₀^{t_max*} P_{rs}(u) du = O(1)  as ε → 0.
    For L = 1 the integral diverges as O(ε⁻¹).

    PROVIDED SOLUTION
    Step 1: Bound the preconditioner term-by-term:
            σ_r^{2(L-a)/L} σ_s^{2(a-1)/L} ≤ max(σ_r,σ_s)^{2(L-1)/L} ≤ σ_1(t)^{2(L-1)/L}.
            So ∫₀^{t_max*} P_{rs}(u) du ≤ L ∫₀^{t_max*} σ_1(u)^{2(L-1)/L} du.
    Step 2: Change variables u ↦ σ_1 using the diagonal ODE of Proposition 6.1:
            σ̇_1 ≥ C λ_1* σ_1^{(2L-1)/L} for some absolute constant C > 0.
            Hence du ≤ dσ_1 / (C λ_1* σ_1^{(2L-1)/L}).
    Step 3: Substitute to get:
            ∫₀^{t_max*} σ_1^{2(L-1)/L} du ≤ (L / (C λ_1*)) ∫_{ε^{1/L}}^{σ_1*} σ_1^{-1/L} dσ_1.
    Step 4: The exponent -1/L > -1 iff L > 1. For L ≥ 2:
            ∫₀^{σ_1*} σ_1^{-1/L} dσ_1 = σ_1*^{1-1/L} / (1-1/L) = O(1).
    Step 5: For L = 1: the integrand is σ_1^{-1}, giving log(σ_1*/ε) → ∞. -/
lemma preconditioner_integral_bounded (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r s : Fin d)
    (sigma_r sigma_s sigma_1 : ℝ → ℝ)
    (t_max : ℝ) (ht_max : 0 < t_max)
    -- Diagonal amplitudes bounded above by σ_1 (the leading amplitude)
    (h_sigma_bound : ∀ t ∈ Set.Icc 0 t_max,
      sigma_r t ≤ sigma_1 t ∧ sigma_s t ≤ sigma_1 t)
    -- σ_1 satisfies the diagonal ODE lower bound
    (h_sigma1_lb : ∀ t ∈ Set.Icc 0 t_max, ∃ C : ℝ, 0 < C ∧
      deriv sigma_1 t ≥ C * projectedCovariance dat eb ⟨0, by omega⟩ * sigma_1 t ^ ((2 * L - 1 : ℝ) / L)) :
    ∃ C : ℝ, 0 < C ∧
      ∫ u in Set.Ioo 0 t_max,
        preconditioner L (sigma_r u) (sigma_s u)
      ≤ C := by
  sorry

/-- Converse of Lemma 7.2: for L = 1, the integral diverges.

    PROVIDED SOLUTION
    Step 1: For L = 1, P_{rs} = σ_r^0 · σ_s^0 = 1 (trivially, since both exponents vanish).
            Actually for L=1 there is only one term a=1: σ_r^0 · σ_s^0 = 1.
    Step 2: From the L=1 diagonal ODE, σ_1(t) grows and reaches σ_1* at time ~ 1/ε.
    Step 3: The integral ∫₀^{1/ε} 1 du = 1/ε → ∞ as ε → 0. -/
lemma preconditioner_integral_diverges_L1 (dat : JEPAData d) (eb : GenEigenbasis dat)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r s : Fin d) (hrs : r ≠ s)
    (sigma_r sigma_s : ℝ → ℝ) :
    -- For L = 1, the integral grows as O(ε⁻¹)
    ∃ C : ℝ, 0 < C ∧
      ∫ u in Set.Ioo 0 (C / epsilon),
        preconditioner 1 (sigma_r u) (sigma_s u)
      ≥ C / epsilon := by
  sorry

/-- **Theorem 7.3 (Off-diagonal bound).**
    For L ≥ 2, under gradient flow from Assumption 4.1:
    |c_{rs}(t)| = O(ε^{1/L})  for all r ≠ s, t ∈ [0, t_max*].

    PROVIDED SOLUTION
    Step 1: From Lemma 7.1, c_{rs} satisfies a linear ODE with O(ε^{(2L-1)/L}) forcing.
    Step 2: Apply the Grönwall inequality to bound the homogeneous part:
            |c_{rs}(t)| ≤ |c_{rs}(0)| exp(κ_{rs} ∫₀ᵗ P_{rs}(u) du)
                          + O(ε^{(2L-1)/L}) · t_max*,
            where κ_{rs} = |ρ_r*(ρ_r* - ρ_s*)| μ_s.
    Step 3: By Lemma 7.2, ∫₀^{t_max*} P_{rs} du = O(1), so the exponential factor is O(1).
    Step 4: Initial condition: c_{rs}(0) = O(ε^{1/L}) from balanced initialisation.
    Step 5: The forcing term contributes O(ε^{(2L-1)/L}) · O(ε^{-1/L}) = O(ε^{2(L-1)/L}).
    Step 6: Therefore |c_{rs}(t)| ≤ O(ε^{1/L}) · O(1) + O(ε^{2(L-1)/L}) = O(ε^{1/L}). -/
theorem offDiag_bound (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r s : Fin d) (hrs : r ≠ s)
    (c_rs sigma_r sigma_s : ℝ → ℝ)
    (t_max : ℝ) (ht_max : 0 < t_max)
    -- Initial off-diagonal amplitude is O(ε^{1/L})
    (h_init : ∃ C₀ : ℝ, 0 < C₀ ∧ |c_rs 0| ≤ C₀ * epsilon ^ ((1 : ℝ) / L))
    -- c_{rs} satisfies the ODE of Lemma 7.1
    (h_ode : ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      |deriv c_rs t
        + preconditioner L (sigma_r t) (sigma_s t)
          * (eb.pairs r).rho * ((eb.pairs r).rho - (eb.pairs s).rho) * (eb.pairs s).mu
          * c_rs t|
      ≤ C * epsilon ^ ((2 * L - 1 : ℝ) / L))
    -- Preconditioner integral is bounded (from Lemma 7.2)
    (h_int_bound : ∃ C : ℝ, 0 < C ∧
      ∫ u in Set.Ioo 0 t_max, preconditioner L (sigma_r u) (sigma_s u) ≤ C) :
    ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      |c_rs t| ≤ C * epsilon ^ ((1 : ℝ) / L) := by
  sorry

/-! ## Section 8: Main Theorem -/

/-- The sine of the angle between a vector v and its projection onto the r-th eigenvector.
    sin∠(v_r(t), v_r*) = ‖c_{rs}‖ / ‖v_r(t)‖ in appropriate norms. -/
noncomputable def sinAngle (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) (r : Fin d) : ℝ :=
  -- Approximation: use off-diagonal amplitude norms relative to diagonal
  -- TODO: make this definition precise using the Σˣˣ-metric
  let sigma_r := diagAmplitude dat eb Wbar r
  let off_sq := ∑ s : Fin d, if s ≠ r then (offDiagAmplitude dat eb Wbar r s) ^ 2 else 0
  Real.sqrt off_sq / (Real.sqrt (sigma_r ^ 2 + off_sq) + 1)  -- TODO: verify formula

/-- **Theorem 8.1 (JEPA ρ*-ordering without simultaneous diagonalisability).**

    Let L ≥ 2. Let ρ₁* > ρ₂* > … > ρ_d* > 0 be the generalised eigenvalues.
    Train the depth-L linear JEPA model by gradient flow from the balanced
    initialisation at scale ε ≪ 1. Then:

    (A) Quasi-static decoder:   ‖V(t) - V_qs(W̄(t))‖ = O(ε^{2(L-1)/L}) → 0.
    (B) Off-diagonal alignment: |c_{rs}(t)| = O(ε^{1/L}) and sin∠(v_r(t), v_r*) = O(ε^{1/L}) → 0.
    (C) Feature ordering:       ρ_r* > ρ_s* ⟹ t̃_r* < t̃_s* for ε sufficiently small.
    (D) Depth threshold:        For L = 1, the ordering theorem is not established
                                (the Grönwall integral diverges).
    (E) JEPA vs. MAE:           When λ_r* = λ_s*, JEPA still orders (t̃_s*/t̃_r* > 1 for ρ_r* > ρ_s*);
                                MAE cannot distinguish the two features.

    PROVIDED SOLUTION
    Step 1 (Part A): Apply Lemma 5.2 (quasi-static decoder approximation).
                     The two-phase argument (Phase A: decoder transient, Phase B: contraction-drift)
                     gives ‖V(t) - V_qs(W̄(t))‖ = O(ε^{2(L-1)/L}).

    Step 2 (Part B, off-diagonal): Combine Lemma 7.1 (off-diagonal ODE), Lemma 7.2
                     (preconditioner integral O(1) for L ≥ 2), and Theorem 7.3 (Grönwall).
                     Initial data c_{rs}(0) = O(ε^{1/L}), integral factor O(1), forcing O(ε^{2(L-1)/L}),
                     conclude |c_{rs}(t)| = O(ε^{1/L}).
                     The sine bound follows from the definition of sinAngle and the amplitude bound.

    Step 3 (Part C, ordering): Apply Proposition 6.1 (diagonal ODE) and Corollary 6.2
                     (critical time formula). With off-diagonal corrections of size
                     O(ε^{2(L-1)/L}|log ε|) subleading to O(ε^{-1/L}), the ordering
                     ρ_r* > ρ_s* ⟹ t̃_r* < t̃_s* follows from critical_time_ordering.

    Step 4 (Part D, depth threshold): By preconditioner_integral_diverges_L1,
                     for L = 1 the Grönwall integral diverges as O(ε⁻¹).
                     The bound |c_{rs}(t)| = O(ε^{1/L}) cannot be established,
                     and the ordering argument breaks down.

    Step 5 (Part E, JEPA vs. MAE): With λ_r* = λ_s*, the critical time ratio from
                     Corollary 6.2 is t̃_s*/t̃_r* = ρ_r*^{2L-2} / ρ_s*^{2L-2} > 1
                     when ρ_r* > ρ_s* and L ≥ 2. For MAE the drive term is V^T Σʸˣ
                     (independent of W̄), so the gradient in mode r is the same for
                     any two features with the same λ* — MAE cannot distinguish them. -/
theorem JEPA_rho_ordering (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    -- Gradient flow from balanced initialisation
    (h_init : BalancedInit d L epsilon)
    -- Encoder product is the composition of layers
    (h_Wbar : ∀ t : ℝ, True)  -- placeholder for gradient flow ODE hypotheses
    :
    -- (A) Quasi-static decoder
    (∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      ‖V t - quasiStaticDecoder dat (Wbar t)‖ ≤ C * epsilon ^ (2 * ((L : ℝ) - 1) / L))
    ∧
    -- (B) Off-diagonal alignment
    (∃ C : ℝ, 0 < C ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
      |offDiagAmplitude dat eb (Wbar t) r s| ≤ C * epsilon ^ ((1 : ℝ) / L))
    ∧
    (∃ C : ℝ, 0 < C ∧ ∀ r : Fin d, ∀ t ∈ Set.Icc 0 t_max,
      sinAngle dat eb (Wbar t) r ≤ C * epsilon ^ ((1 : ℝ) / L))
    ∧
    -- (C) Feature ordering
    (∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon < epsilon_0 →
      ∀ r s : Fin d, (eb.pairs s).rho < (eb.pairs r).rho →
      (L : ℝ) / (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
      < (L : ℝ) / (projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L)))
    ∧
    -- (D) Depth is a sharp threshold: stated as the L=1 divergence (see preconditioner_integral_diverges_L1)
    (L = 1 → ∀ r s : Fin d, r ≠ s →
      ∀ C : ℝ, 0 < C →
      ∃ sigma_r sigma_s : ℝ → ℝ,
        ∫ u in Set.Ioo 0 (C / epsilon), preconditioner 1 (sigma_r u) (sigma_s u) ≥ C / epsilon)
    ∧
    -- (E) JEPA vs. MAE: when λ_r* = λ_s*, JEPA still orders
    (∀ r s : Fin d, r ≠ s →
      projectedCovariance dat eb r = projectedCovariance dat eb s →
      (eb.pairs s).rho < (eb.pairs r).rho →
      (eb.pairs r).rho ^ (2 * L - 2 : ℕ) / (eb.pairs s).rho ^ (2 * L - 2 : ℕ) > 1) := by
  sorry
