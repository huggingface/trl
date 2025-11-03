# Experimental

This directory contains a minimal, clearly separated space for fast iteration on new ideas.

> [!WARNING]
> **Stability contract:** Anything under `trl.experimental` may change or be removed in *any* release (including patch versions) without prior deprecation. Do not rely on these APIs for production workloads.

## Promotion Path (Simple)

1. **Prototype outside the main repo:** Start development in your own fork or a separate repository to iterate quickly.
2. **Experimental inclusion:** Once it’s ready for early users, move the idea into `trl.experimental.<feature>`.
3. **Improve:** Add tests, a short doc/example, and demonstrate the usage.
4. **Promote:** Once the API proves stable and there is clear interest or adoption from the community, move it into `trl.<feature>` (stable module).

## FAQ

**Why not just use branches?**
Because branches are not shipped to users; experimental code inside the package lets early adopters try things and give feedback.

**Can these APIs change or vanish without warning?**
Yes. Anything inside `trl.experimental` can change or disappear in *any* release.

**Should I use this in production?**
Only if you are fine with updating your code quickly when things change.

**Will maintainers promptly fix issues in `trl.experimental`?**
Not necessarily. The experimental module is a playground for new ideas, and maintainers may not prioritize bug fixes or feature requests there. Issues may remain unresolved until (or unless) the feature graduates to the stable API.

**How to silence the runtime notice?**

Use: `export TRL_EXPERIMENTAL_SILENCE=1`.
