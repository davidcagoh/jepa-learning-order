import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic

/-!
# JEPA — Hitting time & Bernoulli machinery (Section 5)

Houses the `hittingTime` definition (used by `SaxeAsymptoticHelpers` and
the Saxe-form analysis in `Corrected.lean`).

History note (session 99, reimagine plan, Phase 2′): this file previously
hosted an inverted-form Bernoulli closed-form solution chain
(`critical_time_formula`, `critical_time_ordering`,
`bernoulli_partial_fractions`, `bernoulli_antideriv_hasDerivAt`,
`jepa_bernoulli_solution`, `jepa_critical_time_diag`) all of which were
deprecated and have been removed.
The Saxe-form analogue lives in `Corrected.lean::saxe_singlepole_asymptotic`
(a single-pole asymptotic, not a Laurent series; this is the right answer
mathematically for the Saxe-form ODE).
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ} (hd : 0 < d)

/-- **Hitting time of a continuous process at threshold θ.**

    First time at which `f t ≥ θ`. Defined as the infimum over the set
    `{t ∈ Set.Icc 0 t_max | f t ≥ θ}`; if the set is empty, defaults to
    `t_max + 1` (an unattainable sentinel).

    Properties (non-negativity, set-nonempty equivalence, csInf equality,
    membership in `Icc`, value at hit, monotone bounds) live in
    `SaxeAsymptoticHelpers.lean`. -/
noncomputable def hittingTime (f : ℝ → ℝ) (θ : ℝ) (t_max : ℝ) : ℝ :=
  sInf ({t ∈ Set.Icc (0 : ℝ) t_max | f t ≥ θ} ∪ {t_max + 1})
