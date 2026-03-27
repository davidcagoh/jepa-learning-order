# Prompt 2 — The Full Recovery Proof
## Task for Claude Cowork

**Files to attach:**
- `JEPA_Proof_Lecture_v2.md` — the completed proof from Prompt 1 (treat every
  result in that document as an established theorem; do not re-prove them)
- `rho_recovery_roadmap.md` — the theorem roadmap (the dependency structure
  for what you are building here)
- `2407_03475v1.pdf` — the original Littwin et al. (2024) arXiv paper
  (reference for notation and the simultaneously-diagonalisable baseline)

**Do not attach** `JEPA_Proof_Lecture_annotated.md` — it records the status of
the earlier formalisation and is not relevant to this task.

---

## Goal

Produce a single self-contained mathematical document
`JEPA_rho_recovery_proof.md` that proves the following:

> **Recovery theorem (informal).** Given a real-world dataset $(x, y)$, one
> can recover the generalised regression coefficients $\rho_r^*$ for all
> features — including their signs — by combining (a) JEPA training dynamics
> (for positive-$\rho$ features) and (b) direct covariance estimation (for
> negative-$\rho$ features), with quantified finite-sample error rates.

This extends the work in `JEPA_Proof_Lecture_v2.md`, which proves only that
JEPA orders features by $\rho^*$ (assuming all $\rho_r^* > 0$). The new
document proves that the $\rho^*$ values themselves are recoverable.

---

## Structure of the Document

Write the document in the following order. Each section heading is a required
deliverable; do not skip or merge sections.

---

### Section 1 — Recap and Standing Assumptions

State concisely (no new proofs) the results inherited from
`JEPA_Proof_Lecture_v2.md` that will be used as black boxes:

- Theorem 8.1 (A): $V(t) \approx V_\mathrm{qs}(\bar W(t))$ up to
  $O(\epsilon^{2(L-1)/L})$.
- Theorem 8.1 (B): $|c_{rs}(t)| = O(\epsilon^{1/L})$ for all $r \neq s$.
- Theorem 8.1 (C): $\rho_r^* > \rho_s^* > 0 \Rightarrow \tilde t_r^* <
  \tilde t_s^*$ for all $\epsilon < \epsilon_0$.
- Corollary 6.2: the critical time formula
  $\tilde t_r^* = \frac{1}{\lambda_r^*}\sum_{n=1}^{2L-1}
  \frac{L}{n\,\rho_r^{*2L-n-1}\epsilon^{n/L}} + \Theta(\log\epsilon)$.
- Proposition 6.1: the diagonal ODE
  $\dot\sigma_r = \sigma_r^{3-1/L}\lambda_r^* - \sigma_r^3\lambda_r^*/\rho_r^*
  + O(\epsilon^{(2L-1)/L})$.

State explicitly that throughout `JEPA_Proof_Lecture_v2.md`, all $\rho_r^*$
are assumed positive. This assumption is relaxed starting in Section 3.

---

### Section 2 — Identifiability: Recovering $\rho_r^*$ from the Training Trajectory

**Context.** Under the assumptions of `JEPA_Proof_Lecture_v2.md` (all
$\rho_r^* > 0$), JEPA training produces critical times $\{\tilde t_r^*\}$.
This section shows that these critical times *identify* the $\rho_r^*$ values,
not just their ordering.

**Theorem 2.1 (Inversion formula).** Let $L \geq 2$ and $\rho_r^* > 0$.
Define the estimator
$$\hat\rho_r(\epsilon) = \left(\frac{L}{\lambda_r^*\,\tilde t_r^*\,
  \epsilon^{1/L}}\right)^{1/(2L-2)}.$$
Then $\hat\rho_r(\epsilon) \to \rho_r^*$ as $\epsilon \to 0$, with
$$|\hat\rho_r(\epsilon) - \rho_r^*| \leq C(L, \rho_r^*)\,\epsilon^{1/L}
  |\log\epsilon|,$$
for an explicit constant $C(L, \rho_r^*)$ depending only on $L$ and $\rho_r^*$.

*Proof strategy:* Invert the leading-order term of the Corollary 6.2 formula.
Show that the subleading terms (the $n \geq 2$ terms and the $\Theta(\log
\epsilon)$ term) contribute at relative order $O(\epsilon^{1/L}|\log\epsilon|)$
to $\hat\rho_r$. Make this bound explicit.

**Note on $\lambda_r^*$.** The formula requires $\lambda_r^* = \rho_r^*\mu_r$
as input. In practice this is estimated from data as
$\hat\lambda_r^* = \hat\rho_r^* \hat\mu_r$ where $\hat\rho_r^*$ and
$\hat\mu_r$ are the $r$-th generalised eigenvalue and the $\Sigma^{xx}$-norm
of the corresponding eigenvector from the sample covariance matrices. Include
this substitution in the statement and prove that the resulting error in
$\hat\lambda_r^*$ propagates to an $O(1/\sqrt{n})$ error in $\hat\rho_r$
(deferred to Section 4 where finite-sample theory is developed).

---

### Section 3 — Signed-$\rho$ Analysis: Relaxing the Positivity Assumption

This section drops the assumption $\rho_r^* > 0$ entirely.

**Setup.** Partition the features into three classes:
- $\mathcal{P} = \{r : \rho_r^* > 0\}$ — *positive features*
- $\mathcal{Z} = \{r : \rho_r^* = 0\}$ — *null features*
- $\mathcal{N} = \{r : \rho_r^* < 0\}$ — *negative features*

**Theorem 3.1 (Signed-$\rho$ ODE analysis).** Replace the assumption
$\rho_r^* > 0$ with $\rho_r^* \in \mathbb{R}$ throughout Proposition 6.1.
Under the same quasi-static and off-diagonal hypotheses (Theorems 8.1 A and B
from `JEPA_Proof_Lecture_v2.md`), the diagonal ODE
$$\dot\sigma_r = \sigma_r^{3-1/L}\lambda_r^* -
  \frac{\sigma_r^3\lambda_r^*}{\rho_r^*} + O(\epsilon^{(2L-1)/L})$$
has the following long-time behaviour:

(a) **Positive features** ($r \in \mathcal{P}$): the ODE has a unique stable
    fixed point at $\sigma_r^* = (\rho_r^*)^{1/2}\mu_r^{1/2} > 0$.
    $\sigma_r(t) \to \sigma_r^*$ monotonically from below, with critical time
    $\tilde t_r^*$ given by Corollary 6.2.

(b) **Null features** ($r \in \mathcal{Z}$): $\lambda_r^* = 0$, so
    $\dot\sigma_r = O(\epsilon^{(2L-1)/L})$. The amplitude stays at
    $\sigma_r(t) = O(\epsilon^{1/L})$ for all $t \in [0, t_{\max}^*]$.

(c) **Negative features** ($r \in \mathcal{N}$): $\lambda_r^* < 0$, so both
    terms in $\dot\sigma_r$ are negative for $\sigma_r > 0$. The ODE has no
    positive fixed point; $\sigma_r(t) \to 0$ monotonically.

Prove each case by analysing the sign of $\dot\sigma_r$ on $(0, \infty)$.

**Theorem 3.2 (Sign identification from training).** Given the encoder
trajectory $\{\sigma_r(t)\}_{t \geq 0}$:

- $r \in \mathcal{P}$ if and only if $\lim_{t\to\infty}\sigma_r(t) > 0$.
- $r \in \mathcal{Z} \cup \mathcal{N}$ if and only if $\sigma_r(t) \to 0$.

Furthermore, the sign of $\rho_r^*$ for features in $\mathcal{N}$ can be
identified directly from the sample covariance as
$\operatorname{sign}(\hat\rho_r^*) = \operatorname{sign}(\hat\lambda_r^*)$,
where $\hat\lambda_r^*$ is the $r$-th generalised eigenvalue of
$\hat\Sigma^{yx}$ in the $\hat\Sigma^{xx}$-metric.

*State and prove that the Grönwall bound of Theorem 7.3 in
`JEPA_Proof_Lecture_v2.md` continues to hold for negative-$\rho$ features,
with the same $O(\epsilon^{1/L})$ bound on $|c_{rs}|$, because the bound
depends only on $|\rho_r^*(\rho_r^* - \rho_s^*)|\mu_s$ and not on the sign.*

---

### Section 4 — Finite-Sample Theory

This section addresses estimation from $n$ i.i.d. samples $(x_i, y_i)$ of
$(x, y) \in \mathbb{R}^d$, replacing the population covariances $\Sigma^{xx}$,
$\Sigma^{yx}$ with sample covariances $\hat\Sigma^{xx}$, $\hat\Sigma^{yx}$.

**Theorem 4.1 (Perturbation of generalised eigenvalues).** Let the fourth
moments of $(x, y)$ be finite. Then:
$$\|\hat\mathcal{R} - \mathcal{R}\|_\mathrm{op} = O_p\!\left(\frac{1}{\sqrt{n}}\right),$$
where $\mathcal{R} = (\Sigma^{xx})^{-1}\Sigma^{yx}$ and $\hat\mathcal{R} =
(\hat\Sigma^{xx})^{-1}\hat\Sigma^{yx}$.
Consequently, by first-order perturbation theory for non-symmetric matrices
(state and cite the appropriate Bauer–Fike or Wedin-type theorem):
$$|\hat\rho_r^* - \rho_r^*| = O_p\!\left(\frac{1}{\sqrt{n}(\rho_r^* -
  \rho_{r+1}^*)}\right),$$
where the gap $\rho_r^* - \rho_{r+1}^*$ appears in the denominator of the
eigenvector perturbation bound.

*Proof strategy:* Write $\hat\mathcal{R} = \mathcal{R} + E$ where
$\|E\|_\mathrm{op} = O_p(1/\sqrt{n})$ by the law of large numbers and the
delta method applied to the map $(\Sigma^{xx}, \Sigma^{yx}) \mapsto
(\Sigma^{xx})^{-1}\Sigma^{yx}$. Apply the Bauer–Fike theorem to bound
eigenvalue perturbations.

**Theorem 4.2 (End-to-end recovery rate).** Under the conditions of Theorems
2.1 and 4.1, the estimator
$$\hat\rho_r = \left(\frac{L}{\hat\lambda_r^*\,\hat t_r^*\,
  \epsilon^{1/L}}\right)^{1/(2L-2)}$$
satisfies, with probability at least $1 - \delta$:
$$|\hat\rho_r - \rho_r^*| \leq C_1(L, \rho_r^*)\,\epsilon^{1/L}|\log\epsilon|
  + \frac{C_2(L, \rho_r^*)}{\sqrt{n}(\rho_r^* - \rho_{r+1}^*)} +
  O\!\left(\frac{\log(1/\delta)}{n}\right)$$
for explicit constants $C_1, C_2$, provided $n \gg d$ and $\epsilon$ is chosen
to balance the two leading terms.

*Proof strategy:* Combine Theorem 2.1 (approximation error from small
$\epsilon$) and Theorem 4.1 (estimation error from finite samples) via a union
bound. Show that the optimal $\epsilon$ (minimising total error) is
$$\epsilon^* \asymp \left(\frac{1}{\sqrt{n}(\rho_r^* - \rho_{r+1}^*)}\right)^L.$$

**Remark on high dimensions.** In the regime $d \sim n$, $\hat\Sigma^{xx}$ is
ill-conditioned. Note (without full proof) that substituting the ridge
estimator $(\hat\Sigma^{xx} + \eta I)^{-1}\hat\Sigma^{yx}$ introduces a bias
of $O(\eta \|\mathcal{R}\|_\mathrm{op})$ but restores the $O_p(1/\sqrt{n})$
variance rate.

---

### Section 5 — Mixed-Sign Ordering and the Full Recovery Statement

**Theorem 5.1 (Mixed-sign feature ordering).** Let $\rho_1^* \geq \cdots \geq
\rho_k^* > 0 > \rho_{k+1}^* \geq \cdots \geq \rho_d^*$. Under gradient flow
from Assumption 4.1:

(a) All $k$ positive features satisfy the ordering of Theorem 8.1 (C) from
    `JEPA_Proof_Lecture_v2.md`: $\rho_r^* > \rho_s^* > 0 \Rightarrow
    \tilde t_r^* < \tilde t_s^*$.
(b) All $d - k$ negative features are suppressed ($\sigma_r(t) \to 0$).
(c) The positive features are fully learned before any negative feature is
    significantly suppressed in the sense that, for all $r \in \mathcal{P}$
    and $s \in \mathcal{N}$, the critical time $\tilde t_r^*$ satisfies
    $\tilde t_r^* \ll \tilde t_s^{\dagger}$, where $\tilde t_s^{\dagger}$ is
    the suppression timescale $O(|\lambda_s^*|^{-1}\epsilon^{1/L-2})$.

*Proof strategy:* Combine Theorem 3.1 (a) and (c), and note that the
suppression timescale for negative features grows as $\epsilon \to 0$ faster
than the learning timescale for positive features.

**Theorem 5.2 (Full recovery theorem).** Combining all preceding results:
Given $n$ i.i.d. samples $(x_i, y_i)$ and a depth-$L \geq 2$ linear JEPA
model trained with initialisation scale $\epsilon = \epsilon^*(n)$:

(a) *Sign*: for every feature $r$, the sign of $\rho_r^*$ is identified
    consistently as $n \to \infty$.
(b) *Magnitude for positive features*: the estimator $\hat\rho_r$ achieves
    the rate of Theorem 4.2.
(c) *Magnitude for negative features*: the estimator $\hat\rho_r =
    \hat\lambda_r^*/\hat\mu_r$ achieves the rate of Theorem 4.1.
(d) *Ordering*: for all pairs $(r, s)$ with $|\rho_r^*| \neq |\rho_s^*|$,
    the recovered values correctly identify the ordering for all $n$
    sufficiently large.

---

## Output Requirements

1. The document must be entirely self-contained: all notation defined,
   all borrowed results explicitly cited by theorem number from
   `JEPA_Proof_Lecture_v2.md`.
2. Every proof must be a complete, step-by-step mathematical argument.
   "Follows from standard arguments" is not acceptable — write out the
   argument.
3. Every asymptotic claim ($O_p$, $O$, $\Theta$) must come with an explicit
   constant or an explicit reference to where the constant arises.
4. The document must be ready to submit to Aristotle for formalisation review.
   In particular, every theorem statement must have hypotheses that are
   non-vacuously constraining (no `∀ t, True`, no existential witnesses of 0).
5. Use the same notation as `JEPA_Proof_Lecture_v2.md` throughout. Do not
   introduce alternative notation for objects already defined there.

---

## Scope Reminder (from the roadmap)

This task covers:

| Layer | Gap | Description |
|---|---|---|
| 2 | 2.2 | Identifiability / inversion formula (Section 2) |
| 3 | 3.1 | Perturbation under sample noise (Section 4) |
| 3 | 3.2 | End-to-end finite-sample rate (Section 4) |
| 4 | 4.1 | Signed-$\rho$ ODE analysis (Section 3) |
| 4 | 4.2 | Signed-$\rho$ recovery (Section 3) |
| 5 | 5.1 | Mixed-sign ordering (Section 5) |

It assumes Layers 1 and 2, Gap 2.1 are complete (handled by Prompt 1 /
`JEPA_Proof_Lecture_v2.md`).
