"""figures/interactive/template.html + figures/data/bench.json -> figures/interactive/index.html (one self-contained page:
the data is inlined, the only network load is d3 7.9.0 from cdnjs). Run by figures/build.sh after build_data.py."""

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data", "bench.json")))
page = open(os.path.join(HERE, "interactive", "template.html")).read()
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
assert page.count("__BENCH_DATA__") == 1
out = os.path.join(HERE, "interactive", "index.html")
with open(out, "w") as f:
    f.write(page.replace("__BENCH_DATA__", blob))
print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
