import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic
import JepaLearningOrder.JEPA.Bernoulli
import JepaLearningOrder.JEPA.DiagAmpODE
import JepaLearningOrder.JEPA.EncoderHelpers
import JepaLearningOrder.JEPA.OffDiagFinal

/-!
# JEPA — Re-export shim

Thin re-export of all six `JepaLearningOrder.JEPA.*` sub-modules. The original
2002-LOC `JEPA.lean` was split during session 95 per the graph-audit
recommendation in [`audits/jepa-learning-order/REPORT-2026-05-23-jepa-split.md`](../../../audits/jepa-learning-order/REPORT-2026-05-23-jepa-split.md).

**Session 95 follow-up (post-split shim audit):** all internal importers
(`BootstrapLemmas`, `MainTheorem`, `PDLowerHelpers`, `LaurentHelpers`,
`SaxeAsymptoticHelpers`, `Corrected`) have been migrated to narrowest-needed
sub-module imports. The shim is now only kept as a convenience meta-export
for the top-level umbrella `JepaLearningOrder.lean` and any external
consumer that prefers the historical single-import form. Internal code
should `import JepaLearningOrder.JEPA.<SubModule>` directly.
-/
