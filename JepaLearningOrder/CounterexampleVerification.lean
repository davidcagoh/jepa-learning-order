/-
Copyright (c) 2026. All rights reserved.
Released under MIT license.

**Intentional orphan** — no importers by design. This file documents the RK4
disproof of the 4th conjunct of `saxe_exact_solution_exists` (session 94) and
exists for the human reader, not the build graph. Tier-1 audits should skip it.

# Numerical counterexample for `saxe_exact_solution_exists`

This file provides a computational verification that the 4th conjunct
(reachability) of `saxe_exact_solution_exists` is **false** for certain
parameter values.

## Parameters
  L = 2, λ = 1, ρ = 1, ε = 0.5, p = 0.99999.

## Hypothesis check
  `h_t_max_reach`: 2 / (1 * 1 * 0.5^{0.5}) = 2√2 ≈ 2.828 ≤ t_max.

## ODE
  f' = 2 · f^{3/2} · (1 − f²)   (Saxe ODE with L=2, λ/ρ = 1)

## Result
  With t_max = 2√2 ≈ 2.828, the ODE solution satisfies
  `f₀(2.828) ≈ 0.99997 < 0.99999 = threshold`.
  The threshold 0.99999 is not reached until `t ≈ 3.11 > t_max`.

## Root cause
  The hypothesis `h_t_max_reach` is independent of `p` and `ρ`. Near the
  equilibrium `ρ^{1/L} = 1`, the ODE speed `F(f) = 2f^{3/2}(1−f²)` vanishes
  as `f → 1`, causing a logarithmic correction `O(log(1/(1−p)))` to the
  hitting time. For `p → 1` this correction is unbounded and exceeds the
  margin provided by the factor of 2 in `h_t_max_reach`.

## Quantitative analysis
  Using the substitution `v = f^{-(L-1)/L}`, the hitting time satisfies:
    T_reach ≤ ε^{-(L-1)/L} / ((L-1) · λ · (1 − p^L))
  While `h_t_max_reach` gives:
    t_max ≥ 2 · ε^{-(L-1)/L} / ((L-1) · λ)
  So `T_reach ≤ t_max` requires `1/(1−p^L) ≤ 2`, i.e., `p^L ≤ 1/2`.
  For `p = 0.99999`, `L = 2`: `p^L = 0.99998 > 1/2`.  ✗
-/

import Mathlib

/-! ### RK4 numerical integration -/

private def rk4Step (rhs : Float → Float) (y : Float) (h : Float) : Float :=
  let k1 := rhs y
  let k2 := rhs (y + h/2 * k1)
  let k3 := rhs (y + h/2 * k2)
  let k4 := rhs (y + h * k3)
  y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

/-- Saxe ODE RHS for L=2, λ=1, ρ=1: f' = 2·f^{3/2}·(1−f²). -/
private def saxeRHS (f : Float) : Float := 2 * f ^ 1.5 * (1 - f * f)

private def integrate (f0 : Float) (dt : Float) (steps : Nat)
    (threshold : Float) (tmax : Float) : String :=
  let rec go (f : Float) (t : Float) (n : Nat) : String :=
    match n with
    | 0 => s!"End: t={t}, f={f}"
    | n + 1 =>
      let f' := rk4Step saxeRHS f dt
      let t' := t + dt
      if f' >= threshold then s!"Threshold {threshold} reached at t ≈ {t'}, f ≈ {f'}"
      else if t' > tmax + 0.001 then
        s!"Threshold {threshold} NOT reached by t_max = {tmax}.  f(t_max) ≈ {f'} < {threshold}"
      else go f' t' n
  go f0 0 steps

/-! ### Verification -/

-- h_t_max_reach lower bound: 2/√0.5 = 2√2 ≈ 2.828427.
#eval (2.0 : Float) / ((0.5 : Float) ^ (0.5 : Float))

-- p = 0.99999: threshold NOT reached by t_max (counterexample).
#eval integrate 0.5 0.0001 40000 0.99999 2.828427

-- p = 0.99999: threshold reached at t ≈ 3.11 > t_max.
#eval integrate 0.5 0.0001 50000 0.99999 5.0

-- p = 0.9999: threshold reached at t ≈ 2.54 < t_max (theorem holds here).
#eval integrate 0.5 0.0001 40000 0.9999 2.828427

-- p = 0.999: threshold reached at t ≈ 1.96 < t_max (theorem holds here).
#eval integrate 0.5 0.0001 40000 0.999 2.828427
