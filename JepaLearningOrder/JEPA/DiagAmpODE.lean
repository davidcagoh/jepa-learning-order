import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic

/-!
# JEPA — Diagonal Amplitude ODE (Section 5)

Previously hosted three inverted-form lemmas (`diagAmp_ODE`,
`bernoulli_laurent_bound`, `actual_critical_time`). All three were deprecated
in session 90 after empirical validation showed the bracket exponent was
inverted; their Saxe-form replacements live in `Corrected.lean`.

Session 99 (reimagine plan, Phase 2′): the deprecated trio is deleted. The
canonical Saxe-form versions in `Corrected.lean` are the current state of
truth. After Phase 2′ completes, those will be moved here (or to a
`JEPA/Saxe.lean` sibling) under their un-suffixed names.

This file is intentionally minimal; the Saxe-form Section-5 content lives in
`Corrected.lean` pending the rename pass.
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ} (hd : 0 < d)
