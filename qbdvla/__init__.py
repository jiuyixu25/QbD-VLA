"""QbD-VLA: task-guided mixed-precision quantization of vision-language-action policies.

The allocation procedure (``search/``), the round-to-nearest quantizer it allocates over (``quant/``), the component
decomposition of a policy (``components.py``), policy adapters (``models/``), benchmark runners (``benchmarks/``) and the
record format of the evaluation cells (``results.py``)."""

from .components import COMPONENTS, ComponentMap, quantize_scope, resolve_scope
from .quant import BASELINE, PRESETS, QuantSpec, parse_spec, quantize_modules

__version__ = "1.0.0"
__all__ = [
    "COMPONENTS",
    "ComponentMap",
    "QuantSpec",
    "BASELINE",
    "PRESETS",
    "parse_spec",
    "quantize_modules",
    "quantize_scope",
    "resolve_scope",
    "__version__",
]
