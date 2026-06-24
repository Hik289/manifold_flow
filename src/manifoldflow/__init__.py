"""ManifoldFlow core implementation.

Public surface kept minimal at EXP_DESIGN stage; the optimizer classes are
filled out in the RUNNING phase. Pure-function geometry helpers are already
implemented because unit tests depend on them.
"""

from .spd_ops import (
    sym,
    fp32_eigh,
    matrix_sqrt,
    matrix_sqrt_inv,
    symexpm,
    symlogm,
    spectral_clip,
)
from .retraction import qr_retract, polar_retract, procrustes_align
from .tangent import decompose_tangent_normal
from .parametrization import manifold_weight

__all__ = [
    "sym",
    "fp32_eigh",
    "matrix_sqrt",
    "matrix_sqrt_inv",
    "symexpm",
    "symlogm",
    "spectral_clip",
    "qr_retract",
    "polar_retract",
    "procrustes_align",
    "decompose_tangent_normal",
    "manifold_weight",
]
