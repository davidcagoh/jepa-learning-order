import Mathlib
import JepaLearningOrder.Lemmas

/-!
# JEPA — Core Definitions (Sections 1, 2)

Foundation types and operators: `JEPAData`, `GenEigenpair`/`GenEigenbasis`,
gradient operators, `matFrobNorm`, amplitudes, and `preconditioner`.
Extracted from `JepaLearningOrder/JEPA.lean` (session 95 split per
`wiki/audits/jepa-learning-order/README.md`).
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

/-- Frobenius norm for matrices. -/
noncomputable def matFrobNorm {n m : ℕ} (M : Matrix (Fin n) (Fin m) ℝ) : ℝ :=
  Real.sqrt (∑ i, ∑ j, (M i j) ^ 2)

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
noncomputable def regressionOperator (dat : JEPAData d) : Matrix (Fin d) (Fin d) ℝ :=
  dat.SigmaXX⁻¹ * dat.SigmaYX

/-- The JEPA loss function.
    ℒ(W̄, V) = ½ tr(V W̄ Σˣˣ W̄ᵀ Vᵀ) - tr(V W̄ Σʸˣ) + ½ tr(W̄ Σʸʸ W̄ᵀ) -/
noncomputable def JEPALoss (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  (1 / 2) * Matrix.trace (V * Wbar * dat.SigmaXX * Wbarᵀ * Vᵀ)
  - Matrix.trace (V * Wbar * dat.SigmaYX)
  + (1 / 2) * Matrix.trace (Wbar * dat.SigmaYY * Wbarᵀ)

/-- The gradient of the JEPA loss with respect to V:
    ∇_V ℒ = V W̄ Σˣˣ W̄ᵀ - W̄ Σʸˣ W̄ᵀ.

    Convention note: this formula uses the matrix Fréchet derivative convention consistent
    with the quasi-static decoder V_qs = W̄ Σʸˣ W̄ᵀ (W̄ Σˣˣ W̄ᵀ)⁻¹ (Definition 5.1).
    Setting gradV = 0 gives V W̄ Σˣˣ W̄ᵀ = W̄ Σʸˣ W̄ᵀ, i.e. V = V_qs. ✓
    The trailing W̄ᵀ factor (vs. the standard (Σʸˣ)ᵀ W̄ᵀ) follows from Littwin et al. (2024)
    Eq. (4), where the StopGrad on the target branch induces an asymmetric gradient. -/
noncomputable def gradV (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  V * Wbar * dat.SigmaXX * Wbarᵀ - Wbar * dat.SigmaYX * Wbarᵀ

/-- The gradient of the JEPA loss with respect to W̄:
    ∇_{W̄} ℒ = Vᵀ (V W̄ Σˣˣ - W̄ Σʸˣ).

    Convention note: consistent with gradV above and with the preconditioned flow
    in Section 2.3. The factor Vᵀ on the left matches Littwin et al. (2024) Eq. (5). -/
noncomputable def gradWbar (dat : JEPAData d)
    (Wbar V : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Vᵀ * (V * Wbar * dat.SigmaXX - Wbar * dat.SigmaYX)

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
  hmu_def : mu = dotProduct v (dat.SigmaXX.mulVec v)

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
    dotProduct (pairs r).v (dat.SigmaXX.mulVec (pairs s).v) = 0

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
  dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs r).v)

/-- Definition 2.3. The off-diagonal amplitude c_{rs}(t) = u_rᵀ W̄(t) v_s* for r ≠ s. -/
noncomputable def offDiagAmplitude (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ) (r s : Fin d) : ℝ :=
  dotProduct (dualBasis dat eb r) (Wbar.mulVec (eb.pairs s).v)

/-- The balanced network preconditioning coefficient P_{rs}(t) for depth L.
    P_{rs} = Σ_{a=1}^{L} σ_r^{2(L-a)/L} · σ_s^{2(a-1)/L}
    where the exponents are real-valued (fractional for L ≥ 2), requiring Real.rpow.
    Note: P_{rr}(σ, σ) = L · σ^{2(L-1)/L} (the Littwin et al. conservation law form). -/
noncomputable def preconditioner (L : ℕ) (sigma_r sigma_s : ℝ) : ℝ :=
  ∑ a : Fin L,
    Real.rpow sigma_r (2 * ((L : ℝ) - ((a.val : ℝ) + 1)) / (L : ℝ))
    * Real.rpow sigma_s (2 * (a.val : ℝ) / (L : ℝ))

