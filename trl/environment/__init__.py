# flake8: noqa
from typing import TYPE_CHECKING
from ..import_utils import _LazyModule

_import_structure = {
    "base_environment": ["TextEnvironment", "TextHistory"],
}

if TYPE_CHECKING:
    from .base_environment import TextEnvironment, TextHistory
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
