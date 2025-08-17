"""Test backward compatibility for SFTConfig parameters."""

import warnings

from trl import SFTConfig


class TestSFTConfigBackwardCompatibility:
    """Test backward compatibility for deprecated parameters in SFTConfig."""

    def test_max_seq_length_backward_compatibility(self):
        """Test that max_seq_length is properly handled for backward compatibility."""

        # Test using deprecated max_seq_length parameter
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = SFTConfig(max_seq_length=512, output_dir="./test")

            # Filter for our specific warning
            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning) and "max_seq_length" in str(warning.message)
            ]

            # Check that a deprecation warning was issued
            assert len(future_warnings) == 1
            assert "deprecated" in str(future_warnings[0].message).lower()

            # Check that the value was properly transferred to max_length
            assert config.max_length == 512

    def test_max_length_no_warning(self):
        """Test that using max_length doesn't produce warnings."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = SFTConfig(max_length=256, output_dir="./test")

            # Filter for our specific warnings
            relevant_warnings = [
                warning
                for warning in w
                if "max_seq_length" in str(warning.message) or "max_length" in str(warning.message)
            ]

            # Check no relevant warnings were issued
            assert len(relevant_warnings) == 0
            assert config.max_length == 256

    def test_both_parameters_provided(self):
        """Test behavior when both max_seq_length and max_length are provided."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = SFTConfig(max_seq_length=512, max_length=256, output_dir="./test")

            # Filter for our specific warnings
            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning) and "max_seq_length" in str(warning.message)
            ]
            user_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, UserWarning)
                and "max_seq_length" in str(warning.message)
                and "max_length" in str(warning.message)
            ]

            # Should have two warnings: deprecation and conflict
            assert len(future_warnings) == 1
            assert len(user_warnings) == 1

            # Should use max_length value (not max_seq_length)
            assert config.max_length == 256
