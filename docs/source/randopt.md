# RandOpt

`RandOpt` provides a lightweight experimental implementation of random parameter-space search, inspired by perturbation
sampling workflows where you evaluate many noisy model variants and keep the best performers.

It is designed as a small utility that can be integrated with existing trainers and custom evaluation loops.

## Experiment script

An end-to-end GSM8K experiment script for RandOpt is available at:

- `examples/scripts/randopt.py`

The script is kept in `examples/` so the package module `trl.experimental.randopt` stays focused on reusable library APIs.

## Core API

[[autodoc]] experimental.randopt.RandOptConfig

[[autodoc]] experimental.randopt.RandOptSearch

[[autodoc]] experimental.randopt.majority_vote
