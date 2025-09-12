"""Experimental namespace for TRL.

This submodule contains unstable or incubating features. Anything here may change
(or be removed) in any release without deprecation. Use at your own risk.

To silence this notice set environment variable TRL_EXPERIMENTAL_SILENCE=1.
"""
from __future__ import annotations

import os
import warnings

if not os.environ.get("TRL_EXPERIMENTAL_SILENCE"):
    warnings.warn(
        "You are importing from 'trl.experimental'. APIs here are unstable and may change or be removed without notice.",
        UserWarning,
        stacklevel=2,
    )

__all__: list[str] = []  # Populated as experiments mature
