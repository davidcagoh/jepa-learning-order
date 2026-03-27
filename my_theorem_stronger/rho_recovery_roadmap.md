# From $\rho^*$-Ordering to $\rho^*$-Recovery: A Complete Theorem Roadmap

**David Goh — March 2026**
*Prepared as a handoff document for Claude Cowork / Aristotle formalisation.*

---

## Purpose of This Document

This document maps every theorem gap that stands between the current state of
the theory and the following concrete goal:

> **Recovery goal.** Given a real-world dataset $(x, y)$ with sample
> covariances $\hat\Sigma^{xx}$, $\hat\Sigma^{yx}$, train a depth-$L \geq 2$
> linear JEPA model and read off a consistent estimator
> $\hat\rho_r$ of each generalised regression coefficient
> $\rho_r^* = \lambda_r^*/\mu_r$ from the training trajectory.

The document is structured as a **dependency chain**: theorems are ordered so
that each one is blocked by exactly those above it. For each gap, the document
states: (1) what is currently in the paper/notes, (2) why it is insufficient,
and (3) the precise statement needed.

It is intended to be passed verbatim to a formal proof assistant workflow
(Lean 4 / Aristotle) together with `JEPA_Proof_Lecture.md`,
`JEPA_Proof_Lecture_annotated.md`, and `2407_03475v1.pdf`.

---

## Background: What the Existing Theory Provides

### Original paper (Littwin et al. 2024, arXiv:2407.03475)

Under **Assumption 4.1** of that paper (simultaneously diagonalisable
$\Sigma^{xx}, \Sigma^{yx}$; balanced orthogonal initialisation; $\rho_i > 0$
for all $i$), Theorem 4.5 proves that the *critical time*

$$t_r^* = \frac{1}{\lambda_r}\sum_{n=1}^{2L-1}
          \frac{L}{n\,\rho_r^{2L-n-1}\,\epsilon^{n/L}} + \Theta(\log\epsilon)$$

is strictly decreasing in $\rho_r$, so JEPA learns features in decreasing
$\rho$-order. The analysis works componentwise in the standard basis because
that basis simultaneously diagonalises both covariance matrices.

### Your proof notes (JEPA\_Proof\_Lecture.md)

The notes extend the result by **dropping simultaneous diagonalisability**.
The correct analytic object is the generalised eigenvector problem
$\Sigma^{yx}\mathbf{v}_r^* = \rho_r^*\Sigma^{xx}\mathbf{v}_r^*$
(Definition 2.2), with amplitudes $\sigma_r(t) =
\mathbf{u}_r^{*\top}\bar W(t)\mathbf{v}_r^*$ and $c_{rs}(t) =
\mathbf{u}_r^{*\top}\bar W(t)\mathbf{v}_s^*$. The gradient decouples exactly
in this basis (Lemma 3.1, **fully proved**), and the off-diagonal amplitudes
are controlled by the Grönwall argument (Theorem 7.3, **fully proved**).

However, as recorded in the annotated file, **two critical components remain
unproved**, and there are **two additional structural assumptions** that must
be relaxed before recovery is possible.

---

## Gap Catalogue

The gaps fall into four layers. Each later layer depends on all earlier ones.

---

### Layer 1 — Complete the existing proof (internal gaps)

These gaps do not require any new mathematical ideas; they are places where the
current Lean formalisation is vacuously satisfied.

---

#### Gap 1.1 — Quasi-static decoder (Lemma 5.2 / Theorem 8.1 A)

**Current state.** The Lean hypothesis reads
```
hV_flow : ∀ t, deriv (fun t => ‖V t - V_qs(Wbar t)‖) t ≤ 0
```
Lean's `deriv` returns 0 when the function is not differentiable, so this
condition is vacuously satisfiable and gives the proof assistant nothing.
Consequently `JEPA_part_A` carries `h_Wbar : ∀ t : ℝ, True` — the encoder
trajectory is completely unconstrained.

**What is needed.** Replace the hypotheses with the actual gradient flow ODEs:

> **Theorem 1.1 (Quasi-static decoder — rigorous statement).**
> Suppose the encoder and decoder satisfy the gradient flow ODEs
> $$\forall t,\quad \operatorname{HasDerivAt}\,\bar W\;
>   \bigl(-\nabla_{\bar W}\mathcal{L}(\bar W(t), V(t))\bigr)\;t,$$
> $$\forall t,\quad \operatorname{HasDerivAt}\,V\;
>   \bigl(-\nabla_{V}\mathcal{L}(\bar W(t), V(t))\bigr)\;t,$$
> with balanced initialisation at scale $\epsilon \ll 1$ and $L \geq 2$. Then
> $$\|V(t) - V_{\mathrm{qs}}(\bar W(t))\| = O(\epsilon^{2(L-1)/L})
>   \quad \text{uniformly on } [0, t_{\max}^*].$$

The proof is the two-phase argument already written in the notes (Phase A:
decoder transient over $O(\epsilon^{-2/L})$; Phase B: contraction–drift
balance). The formalisable ingredients are: (a) explicit solution of the
decoder ODE with $\bar W$ frozen at $\epsilon^{1/L}I$; (b) the Grönwall
inequality applied to $\|\Delta V\|$ in Phase B with contraction rate
$\alpha(t) = O(\epsilon^{2/L})$ and drift rate $O(\epsilon^2)$.

**Dependency.** All subsequent theorems use $V \approx V_{\mathrm{qs}}$ as
an input; nothing downstream can be proved without this.

---

#### Gap 1.2 — Feature ordering (Theorem 8.1 C / `JEPA_rho_ordering` part C)

**Current state.** The Lean existential
```
∃ epsilon_0 > 0, epsilon < epsilon_0 → (ρ_r > ρ_s → t_crit r < t_crit s)
```
was discharged with the witness `epsilon_0 = 0`, making the antecedent
`epsilon < 0` vacuously false. The ordering inequality is never established.

**What is needed.** A direct universal statement:

> **Theorem 1.2 (Feature ordering — rigorous statement).**
> For all pairs $r, s$ with $\rho_r^* > \rho_s^* > 0$, and for all
> $\epsilon$ sufficiently small (i.e. $\epsilon < \epsilon_0$ for an
> *explicitly constructed* $\epsilon_0 > 0$ depending on $\rho_r^*, \rho_s^*,
> \lambda_r^*, \mu_r$), we have
> $$\tilde t_r^* < \tilde t_s^*.$$

The proof follows directly from the closed-form expression for $\tilde t_r^*$
(Corollary 6.2) by showing the leading term $\frac{L}{\lambda_r^*
\rho_r^{*2L-2}\epsilon^{1/L}}$ is strictly decreasing in $\rho_r^*$ when
$\lambda_r^*, \mu_r$ satisfy $\lambda_r^* = \rho_r^* \mu_r$ — which is just
algebra. The key step is to produce a concrete lower bound on
$\tilde t_s^* - \tilde t_r^*$ in terms of $\rho_r^* - \rho_s^*$ and
$\epsilon$.

**Dependency.** Required before any recovery procedure can claim theoretical
justification.

---

### Layer 2 — Extend to the general (non-simultaneously-diagonalisable) case

The notes already do most of this work conceptually. The remaining gap is
making Layer 1 rigorous so that the generalised-eigenbasis analysis is
fully grounded.

---

#### Gap 2.1 — Diagonal ODE under the generalised eigenbasis (Proposition 6.1)

**Current state.** The Lean version has `sigma_r` as an unconstrained
function. The derivation in the notes is informal: it invokes "the Littwin et
al. analysis applies with parameters $(\lambda_r^*, \mu_r, \rho_r^*)$" without
an explicit reduction.

**What is needed.**

> **Theorem 2.1 (Generalised diagonal ODE).** Under Theorem 1.1
> (quasi-static decoder), the diagonal amplitude $\sigma_r(t) =
> \mathbf{u}_r^{*\top}\bar W(t)\mathbf{v}_r^*$ satisfies, to leading
> order in $\epsilon$,
> $$\dot\sigma_r = \sigma_r^{3-1/L}\,\lambda_r^* -
>   \frac{\sigma_r^3\,\lambda_r^*}{\rho_r^*},$$
> where the error term is $O(\epsilon^{(2L-1)/L})$.

The proof reduces to: (i) applying Lemma 3.1 in the generalised eigenbasis to
get the projected gradient; (ii) substituting $V \approx \mathrm{diag}(\rho_r^*)$
(which holds by Theorem 1.1); (iii) applying the Arora et al. (2019) balancedness
lemma to replace the layer-by-layer dynamics with the preconditioned product
dynamics; (iv) verifying that the off-diagonal coupling $c_{rs} = O(\epsilon^{1/L})$
(Theorem 7.3) enters the $\sigma_r$ equation only at order $O(\epsilon^{(2L-1)/L})$.

---

#### Gap 2.2 — Identification of $\rho_r^*$ from the training trajectory

**Current state.** Not present anywhere in the existing paper or notes.
This is the first genuinely new theorem needed for the recovery goal.

**What is needed.**

> **Theorem 2.2 (Identifiability of $\rho^*$ from dynamics).**
> Given the training trajectory $\{\sigma_r(t)\}_{t \geq 0}$ and the
> critical times $\{\tilde t_r^*\}$, define the estimator
> $$\hat\rho_r = \left(\frac{L}{\lambda_r^*\,\tilde t_r^*\,\epsilon^{1/L}}\right)^{1/(2L-2)}.$$
> Then $\hat\rho_r \to \rho_r^*$ as $\epsilon \to 0$, with
> $$|\hat\rho_r - \rho_r^*| = O(\epsilon^{1/L}\,|\log\epsilon|).$$

This theorem converts Corollary 6.2 into an *inversion formula*. The key
steps are: (i) the leading-order term of $\tilde t_r^*$ gives
$\rho_r^* \approx \bigl(L / (\lambda_r^* \tilde t_r^* \epsilon^{1/L})\bigr)^{1/(2L-2)}$;
(ii) the subleading terms (order $\epsilon^{2/L}$ and $\log\epsilon$) bound the
approximation error; (iii) $\lambda_r^* = \rho_r^* \mu_r$ can be estimated
from data (see Layer 3 below), so the formula is computable.

**Remark.** This theorem requires $L \geq 2$ and $\rho_r^* > 0$. The
positivity assumption is still present here; it is relaxed in Layer 4.

---

### Layer 3 — Finite-sample estimation theory

Even with the exact theoretical result of Theorem 2.2, in practice one only
has access to sample covariances $\hat\Sigma^{xx}$, $\hat\Sigma^{yx}$
computed from $n$ i.i.d. observations of $(x, y) \in \mathbb{R}^d$.

---

#### Gap 3.1 — Perturbation of generalised eigenvalues under sample noise

**Current state.** Not addressed anywhere.

**What is needed.**

> **Theorem 3.1 (Stability of $\rho_r^*$ under covariance estimation).**
> Let $\hat\rho_r^*$ be the $r$-th generalised eigenvalue of
> $(\hat\Sigma^{xx})^{-1}\hat\Sigma^{yx}$ computed from $n$ samples.
> Under mild moment conditions on $(x, y)$,
> $$|\hat\rho_r^* - \rho_r^*| = O_p\!\left(\frac{1}{\sqrt{n}}\cdot
>   \frac{1}{\rho_r^* - \rho_{r+1}^*}\right).$$

This is a standard consequence of first-order matrix perturbation theory
(Wedin's theorem or the Davis–Kahan theorem for the non-symmetric case),
combined with the rate $\|\hat\Sigma - \Sigma\| = O_p(1/\sqrt{n})$ from the
law of large numbers. The dependence on the eigengap $\rho_r^* - \rho_{r+1}^*$
is unavoidable and highlights that features with nearly equal $\rho^*$ are
hard to separate.

**Note.** In high dimensions ($d \sim n$), the sample covariance inverse is
ill-conditioned and the $O_p(1/\sqrt{n})$ rate degrades. A ridge-regularised
estimator $(\hat\Sigma^{xx} + \eta I)^{-1}\hat\Sigma^{yx}$ can be substituted,
at the cost of a bias term of order $O(\eta \cdot \|\mathcal{R}\|)$.

---

#### Gap 3.2 — End-to-end finite-sample recovery rate

**Current state.** Not addressed.

**What is needed.**

> **Theorem 3.2 (Finite-sample recovery of $\rho^*$).**
> Train the linear JEPA model on $n$ samples to recover the critical time
> $\hat t_r^*$. Then the estimator
> $$\hat\rho_r = \left(\frac{L}{\hat\lambda_r^*\,\hat t_r^*\,\epsilon^{1/L}}
>               \right)^{1/(2L-2)}$$
> satisfies, with high probability,
> $$|\hat\rho_r - \rho_r^*| \lesssim \frac{C(\rho_r^*, L)}{\sqrt{n}} +
>   O(\epsilon^{1/L}|\log\epsilon|),$$
> where $C(\rho_r^*, L)$ is a problem-dependent constant, provided
> $n \gg d$ and the eigengap $\rho_r^* - \rho_{r+1}^*$ is bounded away from 0.

This combines Theorem 2.2 (approximation error from the dynamics) with
Theorem 3.1 (estimation error from finite samples) via a union bound.

---

### Layer 4 — Relax the positivity assumption on $\rho^*$

This is the most structurally novel layer. The entire existing theory
(paper and notes) assumes $\rho_r^* > 0$ for all $r$. In real data, the
regression coefficient $\rho_r^* = \lambda_r^*/\mu_r$ can be negative
(a feature of $x$ that anti-predicts $y$), zero (a feature of $x$ irrelevant
to $y$), or positive.

---

#### Gap 4.1 — Analysis of the ODE when $\rho_r^* \leq 0$

**Current state.** The diagonal ODE
$$\dot\sigma_r = \sigma_r^{3-1/L}\lambda_r^* - \sigma_r^3\lambda_r^*/\rho_r^*$$
is undefined when $\rho_r^* = 0$ (division by zero) and changes qualitative
character when $\rho_r^* < 0$.

Specifically:
- When $\rho_r^* > 0$: the ODE has a stable fixed point at
  $\sigma_r^* = (\rho_r^*)^{1/2}\mu_r^{1/2} > 0$. The feature is learned.
- When $\rho_r^* = 0$: $\lambda_r^* = 0$, so $\dot\sigma_r = 0$; the feature
  is neither learned nor suppressed.
- When $\rho_r^* < 0$: $\lambda_r^* < 0$, and the fixed point analysis
  changes. The ODE becomes
  $\dot\sigma_r = \sigma_r^{3-1/L}|\lambda_r^*|(-1 - \sigma_r^2/|\rho_r^*|) < 0$
  for $\sigma_r > 0$, so the encoder weight for feature $r$ is *actively
  suppressed* to zero.

**What is needed.**

> **Theorem 4.1 (Signed-$\rho$ ODE analysis).**
> Under the same conditions as Theorem 2.1:
> - If $\rho_r^* > 0$: $\sigma_r(t) \to \sigma_r^* = (\rho_r^*)^{1/2}\mu_r^{1/2}$
>   with critical time $\tilde t_r^* = O(\epsilon^{-1/L})$ as above.
> - If $\rho_r^* = 0$: $\sigma_r(t) = O(\epsilon^{1/L})$ for all $t$ (no learning).
> - If $\rho_r^* < 0$: $\sigma_r(t) \to 0$ monotonically, with suppression
>   timescale $\tilde t_r^{\dagger} = O(|\lambda_r^*|^{-1}\epsilon^{1/L-2})$.

---

#### Gap 4.2 — Recovery of sign and magnitude when $\rho_r^*$ may be negative

**Current state.** Not addressed.

**What is needed.**

> **Theorem 4.2 (Signed-$\rho$ recovery).**
> Given the training trajectory $\{\sigma_r(t)\}$:
>
> (i) *Sign identification.* Feature $r$ has $\rho_r^* > 0$ if and only if
>     $\sigma_r(t) \to \sigma_r^* > 0$; it has $\rho_r^* \leq 0$ if and only if
>     $\sigma_r(t) \to 0$. The sign of $\rho_r^*$ (positive or negative) can
>     be read off from the sign of $\hat\lambda_r^* = \hat\rho_r^* \hat\mu_r$,
>     where $\hat\lambda_r^*$ is the $r$-th generalised eigenvalue of
>     $\hat\Sigma^{yx}$ in the $\hat\Sigma^{xx}$-metric.
>
> (ii) *Magnitude recovery for positive features.* For features with
>     $\rho_r^* > 0$, apply the estimator of Theorem 2.2.
>
> (iii) *Magnitude recovery for negative features.* For features with
>     $\rho_r^* < 0$, the encoder weight $\sigma_r(t)$ decays to zero and
>     the feature is suppressed. The magnitude $|\rho_r^*|$ is *not* directly
>     recoverable from the JEPA trajectory alone; it must be estimated
>     directly from the sample covariances via
>     $\hat\rho_r^* = \hat\lambda_r^*/\hat\mu_r$.

**Implication.** For negative-$\rho$ features, JEPA training does not help —
it actively destroys the signal. Direct covariance estimation (as in Theorem
3.1) is the only route to $\rho_r^*$ for those features.

---

### Layer 5 — Joint ordering under mixed signs

Once the signed analysis is complete, a natural question is whether the full
$|\rho_r^*|$-ordering (or some refinement) can be recovered.

---

#### Gap 5.1 — Ordering theorem under mixed-sign $\rho^*$

**Current state.** Theorem 8.1(C) assumes all $\rho_r^* > 0$ and orders by
$\rho_r^*$ directly.

**What is needed.**

> **Theorem 5.1 (Mixed-sign feature ordering).**
> Let $\rho_1^* \geq \cdots \geq \rho_k^* > 0 > \rho_{k+1}^* \geq \cdots
> \geq \rho_d^*$ be the signed generalised eigenvalues.
> Then:
> (a) All $k$ positive-$\rho$ features are learned by JEPA; their critical
>     times satisfy $\tilde t_r^* < \tilde t_s^*$ iff $\rho_r^* > \rho_s^* > 0$.
> (b) All $d - k$ negative-$\rho$ features are suppressed.
> (c) The positive features are learned *before* any negative feature is fully
>     suppressed, provided $\rho_1^* > \max_r |\rho_r^*|$ for $r > k$.

This extends Theorem 8.1 to the realistic setting.

---

## Summary Table

| Layer | Gap | Depends on | Status |
|---|---|---|---|
| 1 | 1.1 Quasi-static ODE hypothesis | — | ⚠️ Needs rigorous Lean hypothesis |
| 1 | 1.2 Feature ordering non-vacuous | 1.1 | ⚠️ Needs explicit $\epsilon_0 > 0$ |
| 2 | 2.1 Generalised diagonal ODE | 1.1, 1.2 | ⚠️ Lean `sigma_r` unconstrained |
| 2 | 2.2 Identifiability / inversion formula | 2.1 | ❌ Not yet stated |
| 3 | 3.1 Perturbation under sample noise | 2.2 | ❌ Not yet stated |
| 3 | 3.2 End-to-end finite-sample rate | 3.1, 2.2 | ❌ Not yet stated |
| 4 | 4.1 Signed-$\rho$ ODE analysis | 2.1 | ❌ Not yet stated |
| 4 | 4.2 Signed-$\rho$ recovery | 4.1, 3.1 | ❌ Not yet stated |
| 5 | 5.1 Mixed-sign ordering | 4.1, 4.2 | ❌ Not yet stated |

Gaps 1.1 and 1.2 are **proof completion** tasks (the math is already in the
notes, the Lean statements need fixing). Gaps 2.1 onward are **new theorems**
not yet written.

---

## Instructions for Aristotle / Claude Cowork

When reviewing any future submission by David Goh, please check:

1. **For every lemma and theorem**, verify that all hypotheses are *non-trivially*
   constraining — in particular, that no hypothesis is of the form
   `∀ t, True` or vacuously satisfiable via `deriv` on a
   potentially non-differentiable function.

2. **For existential witnesses**, verify that the witness is a *positive*
   quantity genuinely depending on the problem parameters — never `0` or a
   degenerate constant.

3. **For any use of the quasi-static approximation** (i.e. any result that
   invokes $V \approx V_{\mathrm{qs}}$), check that Gap 1.1 has been resolved
   first; without it, all downstream results are ungrounded.

4. **For the recovery goal specifically**, flag any result that does not
   address all five layers above. A proof that only establishes ordering
   (Layer 1–2) without addressing identifiability (Layer 2, Gap 2.2) and
   finite-sample behaviour (Layer 3) is incomplete for the stated goal.

5. **On the positivity assumption**: any theorem whose hypotheses include
   $\rho_r^* > 0$ should be explicitly noted as applying only to
   positive-regression features, and the statement should clarify what happens
   to features with $\rho_r^* \leq 0$.

---

*End of roadmap.*
