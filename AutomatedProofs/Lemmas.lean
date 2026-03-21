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

/-! ## Section 1: Positive-Definite Quadratic Lower Bound -/

/-- **Rayleigh quotient lower bound (Horn-Johnson 4.2.2).**
    For a positive definite real symmetric d×d matrix A, there exists λ > 0 such that
    xᵀAx ≥ λ · ‖x‖² for all x : Fin d → ℝ.

    This is the Rayleigh quotient minimum: λ = min_{x ≠ 0} xᵀAx / xᵀx, which is
    positive because A is positive definite and the unit sphere is compact (finite-dimensional).

    PROVIDED SOLUTION
    Step 1: The function f(x) = xᵀAx = Matrix.dotProduct x (A.mulVec x) is continuous on ℝᵈ.
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
    ∃ λ : ℝ, 0 < λ ∧ ∀ x : Fin d → ℝ,
      Matrix.dotProduct x (A.mulVec x) ≥ λ * Matrix.dotProduct x x := by
  sorry

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
    ∃ λ : ℝ, 0 < λ ∧
      Matrix.trace (Mᵀ * M * A) ≥ λ * Matrix.trace (Mᵀ * M) := by
  obtain ⟨λ, hλ_pos, hλ_bound⟩ := pd_quadratic_lower_bound hd A hA
  use λ, hλ_pos
  -- Tr(MᵀMA) = Σᵢ (MᵀMA)ᵢᵢ = Σᵢ dotProduct (M.col i) (A.mulVec (M.col i))
  -- ≥ Σᵢ λ · dotProduct (M.col i) (M.col i) = λ · Tr(MᵀM)
  sorry

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
    (hβ_nn : ∀ s ∈ Set.Icc 0 T, 0 ≤ β s)
    (hβ_int : IntervalIntegrable β MeasureTheory.volume 0 T)
    (hbound : ∀ t ∈ Set.Icc 0 T,
      u t ≤ c + ∫ s in (0 : ℝ)..t, β s * u s) :
    ∀ t ∈ Set.Icc 0 T,
      u t ≤ c * Real.exp (∫ s in (0 : ℝ)..t, β s) := by
  sorry

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
  sorry
