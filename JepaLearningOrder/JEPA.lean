import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic
import JepaLearningOrder.JEPA.Bernoulli
import JepaLearningOrder.JEPA.DiagAmpODE
import JepaLearningOrder.JEPA.EncoderHelpers
import JepaLearningOrder.JEPA.OffDiagFinal

/-!
# JEPA — Re-export shim

Backwards-compatibility shim. The original 2002-LOC `JEPA.lean` was split into
six sub-modules under `JepaLearningOrder.JEPA.*` during session 95 per the
graph-audit recommendation in `wiki/audits/jepa-learning-order/README.md`.

External importers (`BootstrapLemmas`, `MainTheorem`, `PDLowerHelpers`,
`LaurentHelpers`, `SaxeAsymptoticHelpers`, `Corrected`, the top-level
`JepaLearningOrder.lean`) keep `import JepaLearningOrder.JEPA` working
verbatim — every declaration the old monolith exported is re-exported here
transitively.
-/
