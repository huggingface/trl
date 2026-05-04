# ruff: noqa
"""SFT Trainer compile vs eager — verified input shapes.

Prints actual batch shapes on step 0 to confirm both process same-length sequences.
Uses the exact sft.py launch path via accelerate launch.

Run (2 nodes): TEST_MODE=X srun ... trl/scripts/sft.py [args]
"""
# This is documentation only. Run via the launch scripts below.
