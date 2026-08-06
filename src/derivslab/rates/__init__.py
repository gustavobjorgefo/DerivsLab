"""Interest-rate curve models for the DerivsLab rates package.

Exposes the public surface of the rates package: the ``RateCurve``
behavioral interface and the concrete curves (``FlatRateCurve``, and
whatever is added as the package grows — an interpolated curve
bootstrapped from DI futures, a stochastic-simulation curve). Internal
module layout may change; import from here rather than from the
submodules directly.
"""

from __future__ import annotations

from derivslab.rates.base import RateCurve
from derivslab.rates.flat import FlatRateCurve

__all__ = [
    "FlatRateCurve",
    "RateCurve",
]
