# Experimental Features

The `trl.experimental` namespace provides a minimal, clearly separated space for fast iteration on new ideas.

**Stability contract:** Anything under `trl.experimental` may change or be removed in *any* release (including patch versions) without prior deprecation. Do not rely on these APIs for production workloads.

## Usage

```python
from trl.experimental import *  # not recommended (surface may change)
# Prefer explicit imports once specific symbols are available, e.g.:
# from trl.experimental.newtrainer import NewTrainer
```

To silence the runtime notice:

```bash
export TRL_EXPERIMENTAL_SILENCE=1
```

## Promotion Path (Simple)
1. Start: Develope a new idea in `trl.experimental.<feature>`.
2. Improve: Add at least one test, a short doc/example, and demonstrate the usage.
3. Promote: Once the API feels stable, move it into `trl.<Feature>` (the stable namespace).
4. Transition (optional): We may leave a temporary forwarding import in the experimental spot that warns you. It is later removed.

## FAQ
**Why not just use branches?** Because branches are not shipped to users; experimental code inside the package lets early adopters try things and give feedback.

**Can these APIs change or vanish without warning?** Yes. Anything inside `trl.experimental` can change or disappear in *any* release.

**Should I use this in production?** Only if you are fine with updating your code quickly when things change.
