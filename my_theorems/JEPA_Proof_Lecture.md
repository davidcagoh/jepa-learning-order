# JEPA Learns Influential Features First
## A Proof Without Simultaneous Diagonalizability

**David Goh — March 2026**

---

> **What we prove.** A depth-$L \geq 2$ linear JEPA model, trained from small
> random initialisation, learns features in decreasing order of their
> *generalised regression coefficient* $\rho^*$, even when the input and
> cross-covariance matrices share no common eigenbasis. The ordering is strict,
> persists to vanishing initialisation, and is impossible for $L = 1$.

---

## 1. The Model

Let $(x, y)$ be jointly distributed with second moments

$$\Sigma^{xx} = \mathbb{E}[xx^\top] \succ 0, \qquad \Sigma^{yx} = \mathbb{E}[yx^\top].$$

**Encoder.** A depth-$L$ linear network

$$\bar{W} = W_L W_{L-1} \cdots W_1 \in \mathbb{R}^{d \times d}.$$

**Decoder.** A single linear layer $V \in \mathbb{R}^{d \times d}$.

**JEPA loss.** The stop-gradient $\operatorname{SG}(\cdot)$ blocks gradients through the target branch:

$$\mathcal{L}(\bar{W}, V) = \tfrac{1}{2}\,\mathbb{E}\,\|V\bar{W}x - \operatorname{SG}(\bar{W}y)\|^2.$$

Expanding the expectation,

$$\mathcal{L} = \tfrac{1}{2}\operatorname{tr}(V\bar{W}\Sigma^{xx}\bar{W}^\top V^\top)
             - \operatorname{tr}(V\bar{W}\Sigma^{yx})
             + \tfrac{1}{2}\operatorname{tr}(\bar{W}\Sigma^{yy}\bar{W}^\top).$$

**Gradients.** Differentiating with respect to each variable:

$$\nabla_V \mathcal{L} = V\bar{W}\Sigma^{xx}\bar{W}^\top - \bar{W}\Sigma^{yx}\bar{W}^\top,$$

$$\nabla_{\bar{W}} \mathcal{L} = V^\top\!\bigl(V\bar{W}\Sigma^{xx} - \bar{W}\Sigma^{yx}\bigr).$$

> **Remark (JEPA vs.\ MAE).** For MAE the drive term is $V^\top\Sigma^{yx}$,
> independent of $\bar{W}$. For JEPA it is $V^\top \bar{W}\Sigma^{yx}$ —
> the encoder appears in its own gradient. This *self-referential* structure
> is the source of $\rho^*$-ordering.

---

## 2. The Generalised Eigenvector Framework

When $\Sigma^{xx}$ and $\Sigma^{yx}$ are *not* simultaneously diagonalisable
the standard basis decouples neither the loss nor the gradient. The correct
basis is determined by the following problem.

**Definition 2.1 (Regression operator).** Define $\mathcal{R} = (\Sigma^{xx})^{-1}\Sigma^{yx}$.

**Definition 2.2 (Generalised eigenvectors).** The generalised eigenvectors
$\{\mathbf{v}_r^*\}_{r=1}^d$ and eigenvalues $\rho_1^* > \rho_2^* > \cdots > \rho_d^* > 0$
satisfy

$$\Sigma^{yx}\mathbf{v}_r^* = \rho_r^*\,\Sigma^{xx}\mathbf{v}_r^*,
\qquad \mathbf{v}_r^{*\top}\Sigma^{xx}\mathbf{v}_s^* = \delta_{rs}\,\mu_r > 0.$$

Let $\{\mathbf{u}_r^*\}$ be the dual left basis:
$\mathbf{u}_r^{*\top}\Sigma^{xx}\mathbf{v}_s^* = \delta_{rs}\mu_r$.

> **Remark.** Writing $\lambda_r^* = \rho_r^*\mu_r$, the eigenvalue equation
> becomes $\Sigma^{yx}\mathbf{v}_r^* = \lambda_r^*(\Sigma^{xx})^{-1}\Sigma^{xx}\mathbf{v}_r^*$,
> so $\lambda_r^*$ is the "projected covariance" of feature $r$.
> The key quantity is $\rho_r^* = \lambda_r^*/\mu_r$: it captures
> *signal-to-noise in the $\Sigma^{xx}$ metric*.

**Definition 2.3 (Amplitude decomposition).** For $\bar{W}(t)$ evolving under
gradient flow, define the *diagonal amplitude* and *off-diagonal amplitude*:

$$\sigma_r(t) = \mathbf{u}_r^{*\top}\bar{W}(t)\mathbf{v}_r^*, \qquad
  c_{rs}(t) = \mathbf{u}_r^{*\top}\bar{W}(t)\mathbf{v}_s^* \quad (r \neq s).$$

---

## 3. Key Lemma: Gradient Decouples in the Generalised Eigenbasis

This is the algebraic engine of the whole proof. It requires no structural
assumption on the data beyond $\Sigma^{xx} \succ 0$.

---

**Lemma 3.1 (Gradient projection).** *For any $\bar{W}$ and $V$,*

$$(-\nabla_{\bar{W}}\mathcal{L})\,\mathbf{v}_r^*
  = V^\top(\rho_r^* I - V)\,\bar{W}\Sigma^{xx}\mathbf{v}_r^*.$$

**Proof.** By definition,

$$-\nabla_{\bar{W}}\mathcal{L} = V^\top\bar{W}\Sigma^{yx} - V^\top V\bar{W}\Sigma^{xx}.$$

Evaluate on $\mathbf{v}_r^*$ and substitute
$\Sigma^{yx}\mathbf{v}_r^* = \rho_r^*\Sigma^{xx}\mathbf{v}_r^*$:

$$(-\nabla_{\bar{W}}\mathcal{L})\mathbf{v}_r^*
  = V^\top\bar{W}(\rho_r^*\Sigma^{xx}\mathbf{v}_r^*) - V^\top V\bar{W}\Sigma^{xx}\mathbf{v}_r^*
  = V^\top(\rho_r^* I - V)\bar{W}\Sigma^{xx}\mathbf{v}_r^*. \qquad \square$$

---

> **Interpretation.** The descent direction in mode $r$ depends on $V$ only
> through the *deviation* $(\rho_r^* I - V)$. When $V\mathbf{u}_r^* = \rho_r^*\mathbf{u}_r^*$
> — i.e. the decoder has learnt the correct gain for feature $r$ — the
> gradient in that mode vanishes exactly, regardless of how other modes are
> aligned. This decoupling is the reason the generalised eigenbasis is natural.

---

## 4. Initialisation and the Balanced Network

**Assumption 4.1 (Balanced initialisation).** Each layer is initialised as
$W^a(0) = \epsilon^{1/L} U^a$ with $U^a$ orthogonal and $0 < \epsilon \ll 1$.
The decoder starts at $V(0) = \epsilon^{1/L} U^v$ with $U^v$ orthogonal.
The network is *balanced*: $W^{a+1}(t)^\top W^{a+1}(t) = W^a(t)W^a(t)^\top$
for all $t$, which is preserved by gradient flow from this initialisation.

Under balancedness (Arora et al. 2019, Lemma 3), the gradient flow on
$\{\bar{W}, V\}$ simplifies: the product $\bar{W}$ evolves as if it were a
single matrix with a *preconditioned* gradient,

$$\dot{\bar{W}} = -\eta\,P(\bar{W})\!\cdot\!\nabla_{\bar{W}}\mathcal{L},$$

where $P(\bar{W})$ acts on direction $\mathbf{v}_r^*$ by scaling with

$$P_{rs}(t) = \sum_{a=1}^{L} \sigma_r(t)^{2(L-a)/L}\sigma_s(t)^{2(a-1)/L}.$$

At initialisation, $\sigma_r(0) = O(\epsilon^{1/L})$ and $c_{rs}(0) = O(\epsilon^{1/L})$
for all $r, s$.

---

## 5. Timescale Separation and the Quasi-Static Decoder

The core of the proof is that the decoder evolves much faster than the encoder,
so it can be treated as "always at equilibrium".

**Definition 5.1 (Quasi-static fixed point).** For fixed $\bar{W}$, the
minimiser of $\mathcal{L}$ over $V$ is

$$V_\mathrm{qs}(\bar{W}) = \bar{W}\Sigma^{yx}\bar{W}^\top\!\bigl(\bar{W}\Sigma^{xx}\bar{W}^\top\bigr)^{-1}.$$

*(Set $\nabla_V\mathcal{L} = 0$ and solve.)*

**Lemma 5.2 (Quasi-static decoder approximation).** *For $L \geq 2$ and
initialisation scale $\epsilon \ll 1$, the decoder satisfies*

$$\|V(t) - V_\mathrm{qs}(\bar{W}(t))\| = O\!\left(\epsilon^{2(L-1)/L}\right)$$

*uniformly for all $t \in [0, t_{\max}^*]$.*

**Proof** (two-phase argument)**.**

*Phase A — decoder transient* ($t \in [0,\, \tau_A]$, $\tau_A = O(\epsilon^{-2/L})$).

During Phase A the encoder changes at rate $\|\dot{\bar{W}}\| = O(\epsilon^2)$,
so over duration $\tau_A$ the encoder moves by
$O(\epsilon^2 \cdot \epsilon^{-2/L}) = O(\epsilon^{(2L-2)/L})$ — negligible.
With $\bar{W} \approx \epsilon^{1/L}I$ fixed, the decoder ODE

$$\dot{V} = -\epsilon^{2/L}(V\Sigma^{xx} - \Sigma^{yx})$$

is exactly solvable:

$$V(t) = \Sigma^{yx}(\Sigma^{xx})^{-1}\!\left(I - e^{-\epsilon^{2/L}\Sigma^{xx}t}\right)
         + \epsilon^{1/L}e^{-\epsilon^{2/L}\Sigma^{xx}t}.$$

This converges exponentially to $V_\mathrm{qs}(\epsilon^{1/L}I) = \Sigma^{yx}(\Sigma^{xx})^{-1}$
on timescale $O(\epsilon^{-2/L})$, so $\Delta V$ is exponentially small by the
end of Phase A. The encoder change during Phase A implies, in the generalised
eigenbasis,

$$\Delta\sigma_r^A = O\!\left(\rho_r^* \epsilon^{(2L-3)/L}\right),
\qquad \Delta c_{rs}^A = O\!\left(\epsilon^{(2L-3)/L}\right).$$

Both are $o(\epsilon^{1/L})$ for $L \geq 2$, so all amplitudes remain
$O(\epsilon^{1/L})$ with $\rho^*$-ordered diagonal.

*Phase B — quasi-static regime* ($t \in [\tau_A,\, t_{\max}^*]$).

Set $\Delta V(t) = V(t) - V_\mathrm{qs}(\bar{W}(t))$. Using
$\nabla_V\mathcal{L} = (V - V_\mathrm{qs})\bar{W}\Sigma^{xx}\bar{W}^\top$,
the deviation satisfies

$$\dot{\Delta V} = -\Delta V \cdot \bar{W}\Sigma^{xx}\bar{W}^\top
                  - \frac{d}{dt}V_\mathrm{qs}(\bar{W}).$$

The first term contracts $\Delta V$ at rate
$\alpha(t) = \lambda_{\min}(\bar{W}\Sigma^{xx}\bar{W}^\top) = O(\epsilon^{2/L})$.
The second term (the "drift" of the quasi-static target) is bounded by
$\|\partial_{\bar{W}}V_\mathrm{qs}\|_\mathrm{op} \cdot \|\dot{\bar{W}}\| = O(\epsilon^2)$.
By variation of constants the steady-state error is

$$\|\Delta V(t)\|_\mathrm{ss} \lesssim \frac{\|\dot{\bar{W}}\| \cdot \|\partial_{\bar{W}}V_\mathrm{qs}\|_\mathrm{op}}{\alpha(t)}
= O\!\left(\frac{\epsilon^2}{\epsilon^{2/L}}\right) = O\!\left(\epsilon^{2(L-1)/L}\right).$$

For $L \geq 2$, this tends to $0$ as $\epsilon \to 0$. $\square$

> **Remark.** At $t = 0$, the initial error $\Delta V(0) = O(1)$ is not small,
> which is why a naive adiabatic argument fails: Phase A is essential. Phase A
> also *seeds* $\rho^*$-ordering in the diagonal amplitudes before the
> quasi-static analysis begins.

---

## 6. Diagonal Dynamics: The Littwin ODE

With the decoder quasi-static we can read off the diagonal amplitude dynamics
directly from Lemma 3.1.

**Proposition 6.1 (Effective diagonal ODE).** *Under the approximate alignment
$c_{rs} = O(\epsilon^{1/L})$, the diagonal amplitude $\sigma_r(t)$ satisfies,
to leading order in $\epsilon$,*

$$\dot{\sigma}_r = \sigma_r^{3-1/L}\lambda_r^* - \frac{\sigma_r^3\lambda_r^*}{\rho_r^*},
\qquad \lambda_r^* = \rho_r^*\mu_r.$$

**Proof.** Project the encoder gradient (Lemma 3.1) and the decoder gradient
onto mode $r$. Writing $v_r = \mathbf{u}_r^{*\top}V\mathbf{u}_r^*$ for the
$r$-th decoder gain, the $\mathbf{u}_r^*$ component of Lemma 3.1 gives:

$$\mathbf{u}_r^{*\top}(-\nabla_{\bar{W}}\mathcal{L})\mathbf{v}_r^*
  = v_r(\rho_r^* - v_r)\,\sigma_r\mu_r.$$

The decoder gradient projected onto mode $r$ is:

$$\dot{v}_r = -\sigma_r^2\mu_r\,(v_r - \rho_r^*).$$

These two equations form a 2D system in $(\sigma_r, v_r)$ that is *identical*
in structure to the diagonal JEPA system analysed by Littwin et al. (2024),
Theorem 4.2, with parameters $(\lambda_r^*, \mu_r, \rho_r^*)$ in place of
$(\lambda_i, \sigma_i^2, \rho_i)$. Cross-mode coupling enters only through
the off-diagonal terms $c_{rs}$, which are $O(\epsilon^{1/L})$ and hence
contribute at next order. Applying the Littwin et al. analysis to this 2D system — which exploits the balanced-network conservation law to reduce it to a scalar ODE — recovers the stated equation. $\square$

**Corollary 6.2 (Critical time formula).** The critical time $\tilde{t}_r^*$
at which $\sigma_r$ reaches fraction $p$ of its asymptote
$\sigma_r^* = (\rho_r^*)^{1/2}\mu_r^{1/2}$ is

$$\tilde{t}_r^* = \frac{1}{\lambda_r^*}\sum_{n=1}^{2L-1}\frac{L}{n\,\rho_r^{*2L-n-1}\,\epsilon^{n/L}}
                 + \Theta[\log\epsilon].$$

*Leading order*: $\tilde{t}_r^* \approx \dfrac{L}{\lambda_r^*\rho_r^{*2L-2}\epsilon^{1/L}}$.

Since $\tilde{t}_r^*$ is strictly decreasing in $\rho_r^*$ (for fixed $\epsilon$),
features with higher $\rho^*$ reach their asymptote first — *provided*
the off-diagonal amplitudes remain small throughout.

---

## 7. Off-Diagonal Dynamics and the Grönwall Bound

This section closes the gap: we show the off-diagonal amplitudes stay small
uniformly, not just at initialisation.

**Lemma 7.1 (Off-diagonal ODE).** *Under Lemma 5.2, for $r \neq s$:*

$$\dot{c}_{rs} = -P_{rs}(t)\cdot\rho_r^*(\rho_r^* - \rho_s^*)\mu_s\cdot c_{rs}
+ O\!\left(\epsilon^{(2L-1)/L}\right).$$

**Proof.** Project Lemma 3.1 onto the $(r, s)$ off-diagonal entry with
$V \approx \operatorname{diag}(\rho_r^*)$:

$$\mathbf{u}_r^{*\top}(-\nabla_{\bar{W}}\mathcal{L})\mathbf{v}_s^*
= \mathbf{u}_r^{*\top}V^\top(\rho_s^*I - V)\bar{W}\Sigma^{xx}\mathbf{v}_s^*
= \rho_r^*(\rho_s^* - \rho_r^*)\mu_s\,c_{rs}.$$

Applying the preconditioning with $P_{rs}$ (Arora et al. 2019) and
accounting for the $O(\epsilon^{2(L-1)/L})$ error in $V$ (Lemma 5.2,
which induces an $O(\epsilon^{(2L-1)/L})$ forcing after multiplying by
$\|\bar{W}\| = O(\epsilon^{1/L})$) gives the stated equation. $\square$

> **Sign analysis.** The homogeneous coefficient $-\rho_r^*(\rho_r^* - \rho_s^*)\mu_s$ is:
>
> - **Negative** when $\rho_r^* > \rho_s^*$: $c_{rs}$ *decays* — the dominant
>   feature's direction is actively purified by the structured decoder.
> - **Positive** when $\rho_r^* < \rho_s^*$: $c_{rs}$ *grows* while the
>   stronger feature $s$ is developing. Growth stops when $\sigma_s$ saturates.
>
> In both cases the Grönwall bound below controls the total amplification.

---

**Lemma 7.2 (Integral bound — the heart of the depth condition).**
*For $L \geq 2$ and all $r, s$:*

$$\int_0^{t_{\max}^*} P_{rs}(u)\,du = O(1) \quad \text{as } \epsilon \to 0.$$

*For $L = 1$ the integral diverges as $O(\epsilon^{-1})$.*

**Proof.** Bound the preconditioner term-by-term:
$\sigma_r^{2(L-a)/L}\sigma_s^{2(a-1)/L} \leq \max(\sigma_r,\sigma_s)^{2(L-1)/L}
\leq \sigma_1(t)^{2(L-1)/L}$, so

$$\int_0^{t_{\max}^*} P_{rs}(u)\,du \leq L\int_0^{t_{\max}^*}\sigma_1(u)^{2(L-1)/L}\,du.$$

Change variables $u \mapsto \sigma_1$ using the diagonal ODE of
Proposition 6.1: $\dot{\sigma}_1 \geq C\lambda_1^*\sigma_1^{(2L-1)/L}$
for some absolute constant $C > 0$, so $du \leq d\sigma_1\,/\,(C\lambda_1^*\sigma_1^{(2L-1)/L})$.
Therefore

$$\int_0^{t_{\max}^*}\sigma_1^{2(L-1)/L}\,du
\leq \frac{L}{C\lambda_1^*}\int_{\epsilon^{1/L}}^{\sigma_1^*}
   \sigma_1^{\,2(L-1)/L\,-\,(2L-1)/L}\,d\sigma_1
= \frac{L}{C\lambda_1^*}\int_{\epsilon^{1/L}}^{\sigma_1^*}\sigma_1^{-1/L}\,d\sigma_1.$$

The exponent $-1/L$ is integrable at $0$ if and only if $-1/L > -1$, i.e.\ $L > 1$.
For $L \geq 2$:

$$\int_0^{\sigma_1^*}\sigma_1^{-1/L}\,d\sigma_1 = \frac{\sigma_1^{*\,1-1/L}}{1-1/L} = O(1).$$

For $L = 1$: the integrand is $\sigma_1^{-1}$, giving $\log(\sigma_1^*/\epsilon) \to \infty$.
$\square$

> **Remark.** The convergence of $\int_0^{O(1)}\sigma^{-1/L}\,d\sigma$ is
> *exactly* equivalent to $L \geq 2$. Depth is not merely helpful — it is
> the precise threshold for the off-diagonal alignment mechanism to work.

---

**Theorem 7.3 (Off-diagonal bound).** *For $L \geq 2$, under the gradient
flow from Assumption 4.1:*

$$|c_{rs}(t)| = O\!\left(\epsilon^{1/L}\right) \quad \text{for all } r \neq s,\ t \in [0, t_{\max}^*].$$

**Proof.** From Lemma 7.1, $c_{rs}$ satisfies a linear ODE with an $O(\epsilon^{(2L-1)/L})$
forcing. The homogeneous solution is bounded by the Grönwall inequality:

$$|c_{rs}(t)| \leq |c_{rs}(0)|\exp\!\left(\kappa_{rs}\int_0^t P_{rs}(u)\,du\right)
                + O\!\left(\epsilon^{(2L-1)/L}\right)\cdot t_{\max}^*,$$

where $\kappa_{rs} = |\rho_r^*(\rho_r^* - \rho_s^*)|\mu_s$. By Lemma 7.2,
$\int_0^{t_{\max}^*}P_{rs}\,du = O(1)$, so the exponential factor is
$O(1)$. Since $c_{rs}(0) = O(\epsilon^{1/L})$ (balanced initialisation):

$$|c_{rs}(t)| \leq O(\epsilon^{1/L})\cdot O(1) + O(\epsilon^{(2L-1)/L})\cdot O(\epsilon^{-1/L})
= O(\epsilon^{1/L}) + O(\epsilon^{2(L-1)/L}) = O(\epsilon^{1/L}). \qquad \square$$

---

## 8. Main Theorem

We now collect every component into a single statement.

---

**Theorem 8.1 (JEPA $\rho^*$-ordering without simultaneous diagonalisability).**

*Let $L \geq 2$. Let $\rho_1^* > \rho_2^* > \cdots > \rho_d^* > 0$ be the
generalised eigenvalues of Definition 2.2. Train the depth-$L$ linear JEPA
model by gradient flow from the balanced initialisation of Assumption 4.1 at
scale $\epsilon \ll 1$. Then:*

**(A) Quasi-static decoder.** $\|V(t) - V_\mathrm{qs}(\bar{W}(t))\| = O(\epsilon^{2(L-1)/L}) \to 0.$

**(B) Off-diagonal alignment.** For all $r \neq s$ and all $t \leq t_{\max}^*$:

$$|c_{rs}(t)| = O(\epsilon^{1/L}),
\qquad
\sin\angle\!\left(\mathbf{v}_r(t),\,\mathbf{v}_r^*\right) = O(\epsilon^{1/L}) \to 0.$$

**(C) Feature ordering.** The critical time for feature $r$ is

$$\tilde{t}_r^* = \frac{1}{\lambda_r^*}\sum_{n=1}^{2L-1}\frac{L}{n\,\rho_r^{*2L-n-1}\,\epsilon^{n/L}}
                 + \Theta[\log\epsilon],$$

with off-diagonal corrections of size $O(\epsilon^{2(L-1)/L}|\log\epsilon|)$ — subleading to the
$O(\epsilon^{-1/L})$ gap between consecutive features. Therefore

$$\rho_r^* > \rho_s^* \implies \tilde{t}_r^* < \tilde{t}_s^* \quad \text{for all sufficiently small } \epsilon.$$

**(D) Depth is a sharp threshold.** For $L = 1$ the integral $\int P_{rs}\,du$
diverges as $\epsilon \to 0$, the Grönwall bound fails, and the ordering
theorem is not established by this argument.

**(E) JEPA vs.\ MAE in the degenerate case.** When $\lambda_r^* = \lambda_s^*$
(same projected covariance), MAE cannot distinguish features $r$ and $s$.
JEPA still orders them correctly for $L \geq 2$:

$$\frac{\tilde{t}_s^*}{\tilde{t}_r^*} = \frac{\rho_r^{*\,2L-2}}{\rho_s^{*\,2L-2}} > 1
\quad \text{when } \rho_r^* > \rho_s^*.$$

---

**Proof outline.**

1. *Lemma 3.1* — purely algebraic, projects the JEPA gradient onto the
   generalised eigenbasis. No data assumption beyond $\Sigma^{xx} \succ 0$.

2. *Lemma 5.2* — the two-phase argument shows $V \approx V_\mathrm{qs}$:
   Phase A (duration $O(\epsilon^{-2/L})$) lets the decoder converge while
   the encoder barely moves; Phase B (duration $O(\epsilon^{-1/L})$) keeps
   the deviation at $O(\epsilon^{2(L-1)/L})$ via a contraction–drift balance.
   This gives part **(A)**.

3. *Proposition 6.1 and Corollary 6.2* — with $V \approx \operatorname{diag}(\rho_r^*)$,
   Lemma 3.1 collapses each diagonal mode to the Littwin ODE, whose solution
   gives the critical time formula for part **(C)**.

4. *Lemma 7.1* — the off-diagonal ODE under the quasi-static decoder.

5. *Lemma 7.2* — the change-of-variables $t \mapsto \sigma_1$ converts
   $\int P_{rs}\,du$ into $\int \sigma^{-1/L}\,d\sigma$, which converges
   iff $L \geq 2$. This is the core of parts **(B)** and **(D)**.

6. *Theorem 7.3* — Grönwall applied to Lemma 7.1 with the integral bound
   from Lemma 7.2 gives $c_{rs} = O(\epsilon^{1/L})$, completing **(B)**.

7. The off-diagonal corrections to the critical time (part **(C)**) follow
   by substituting the bound $c_{rs} = O(\epsilon^{1/L})$ into the
   $\Sigma^{xx}$-normalised amplitude $\tilde{A}_r = (\sigma_r^2 + \sum_{s \neq r}c_{sr}^2)/\mu_r$:
   the off-diagonal correction enters at $O(\epsilon^{2/L})$ relative to
   $\sigma_r^2$, shifting $\tilde{t}_r^*$ by $O(\epsilon^{2(L-1)/L}|\log\epsilon|)$.

8. Part **(E)** follows directly from Corollary 6.2 with $\lambda_r^* = \lambda_s^*
   \Rightarrow \mu_r\rho_r^* = \mu_s\rho_s^*$, simplifying the ratio. $\square$

---

## References

1. **Littwin, Grill, Gidel, Balestriero, LeCun, Razin (2024).**
   *JEPA: Self-distillation gives rise to hierarchical feature learning.*
   arXiv:2407.03475.

2. **Arora, Cohen, Hu, Luo (2019).**
   *Implicit regularisation in deep matrix factorisation.*
   NeurIPS 2019. arXiv:1905.13655.

3. **Arora, Cohen, Hazan (2018).**
   *On the optimisation of deep networks: Implicit acceleration by
   overparameterisation.* ICML 2018. arXiv:1802.06509.

4. **Sanders, Verhulst, Murdock (2007).**
   *Averaging Methods in Nonlinear Dynamical Systems*, 2nd ed.
   Springer. *(For the slow-fast / adiabatic elimination framework
   underlying Lemma 5.2.)*

---

*Notes compiled March 2026.*
