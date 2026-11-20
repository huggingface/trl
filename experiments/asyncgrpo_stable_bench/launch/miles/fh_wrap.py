"""Run a script with periodic Python stack dumps: fh_wrap.py <interval_s> <script> [args...].
Used for the HF->torch_dist converter, which can hang silently and can't be inspected with py-spy (no ptrace)."""

import faulthandler
import runpy
import sys


faulthandler.dump_traceback_later(int(sys.argv[1]), repeat=True)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
