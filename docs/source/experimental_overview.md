# Experimental

This directory contains a minimal, clearly separated space for fast iteration on new ideas.

> [!WARNING]
> **Stability contract:** Anything under `trl.experimental` may change or be removed in *any* release (including patch versions) without prior deprecation. Do not rely on these APIs for production workloads.

## Promotion Path (Simple)

1. **Prototype outside the main repo:** Start development in your own fork or a separate repository to iterate quickly.
2. **Experimental inclusion:** Once it’s ready for early users, move the idea into `trl.experimental.<feature>`.
3. **Improve:** Add tests, a short doc/example, and demonstrate the usage.
4. **Promote:** Once the API proves stable and there is clear interest or adoption from the community, move it into `trl.<feature>` (stable module).

## Removal

Experimental is a staging area, not long-term storage: a feature is either promoted or removed. Removal needs no deprecation cycle, as stated in the stability contract above. The code remains available in the git history, and the [paper index](paper_index) entry stays, pointing at the last release that shipped the implementation.

No threshold triggers a removal. It is a judgment call, made in the removal pull request with the numbers in front of everyone. What that judgment weighs:

- **Usage.** Telemetry, Hub tags, and replies on the pull request or on a discussion thread. None of them is conclusive on its own, but all of them quiet is a strong signal.
- **External issues and pull requests that come from running the feature**, as opposed to reports produced by scanning the code.
- **Whether a stable trainer already covers it.** If one does, that outweighs usage.
- **Cost.** CI time, maintenance, triage load.
- **Downstream consumers.** Libraries that import it.
- **Owner.** Someone who wants it promoted and is willing to do the work.
- **Age.** Whether it has been available long enough for the community to find it and try it.

## FAQ

**Why not just use branches?**
Because branches are not shipped to users; experimental code inside the package lets early adopters try things and give feedback.

**Can these APIs change or vanish without warning?**
Yes. Anything inside `trl.experimental` can change or disappear in *any* release.

**Should I use this in production?**
Only if you are fine with updating your code quickly when things change.

**Will maintainers promptly fix issues in `trl.experimental`?**
Not necessarily. The experimental module is a playground for new ideas, and maintainers may not prioritize bug fixes or feature requests there. Issues may remain unresolved until (or unless) the feature graduates to the stable API.

**I contributed a paper implementation. Can it be removed?**
Yes. Anything under `trl.experimental` can be removed, paper implementations included. If that happens, the [paper index](paper_index) entry stays and points at the last release that shipped the code.

**How to silence the runtime notice?**

Use: `export TRL_EXPERIMENTAL_SILENCE=1`.
