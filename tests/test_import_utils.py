# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.metadata
import warnings
from unittest.mock import patch

import pytest

import trl.import_utils as import_utils
from trl.import_utils import _get_package_distribution_mapping, _is_package_available


def test_lazy_package_distribution_mapping():
    # Calling _get_package_distribution_mapping returns mapping and caches result
    mapping1 = _get_package_distribution_mapping()
    mapping2 = _get_package_distribution_mapping()
    assert isinstance(mapping1, dict)
    assert mapping1 is mapping2


def test_package_distribution_mapping_deprecated_attribute():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        mapping = import_utils.PACKAGE_DISTRIBUTION_MAPPING
        assert isinstance(mapping, dict)
        assert len(recorded) == 1
        assert issubclass(recorded[0].category, FutureWarning)
        assert "PACKAGE_DISTRIBUTION_MAPPING is deprecated" in str(recorded[0].message)


def test_getattr_unknown_attribute():
    with pytest.raises(AttributeError, match="module 'trl.import_utils' has no attribute 'NON_EXISTENT_ATTRIBUTE'"):
        _ = import_utils.NON_EXISTENT_ATTRIBUTE


def test_is_package_available_standard_packages():
    # Package that exists without version
    assert _is_package_available("pytest") is True

    # Package that exists with version
    exists, version = _is_package_available("pytest", return_version=True)
    assert exists is True
    assert version != "N/A"

    # Package that does not exist
    assert _is_package_available("non_existent_fake_package_xyz123") is False
    exists, version = _is_package_available("non_existent_fake_package_xyz123", return_version=True)
    assert exists is False
    assert version == "N/A"


def test_is_package_available_fast_path_bypasses_distribution_mapping():
    # Fast path should resolve directly via importlib.metadata.version without invoking _get_package_distribution_mapping
    with patch("trl.import_utils._get_package_distribution_mapping") as mock_mapping:
        exists, version = _is_package_available("pytest", return_version=True)
        assert exists is True
        assert version != "N/A"
        mock_mapping.assert_not_called()


def test_is_package_available_fallback_to_distribution_mapping():
    # When distribution name does not match pkg_name directly, it falls back to mapping
    def fake_version(dist):
        if dist == "pillow":
            return "10.0.0"
        raise importlib.metadata.PackageNotFoundError(dist)

    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", side_effect=fake_version),
        patch(
            "trl.import_utils._get_package_distribution_mapping",
            return_value={"PIL": ["pillow"]},
        ),
    ):
        exists, version = _is_package_available("PIL", return_version=True)
        assert exists is True
        assert version == "10.0.0"
