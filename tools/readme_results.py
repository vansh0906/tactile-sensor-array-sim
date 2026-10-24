"""Copy results.txt into README.md between the results markers.

    python run_all.py | tee results.txt
    python tools/readme_results.py
"""
from pathlib import Path

START, END = "<!-- results:start -->", "<!-- results:end -->"
out = Path("results.txt").read_text().strip()
table = out[out.index("ID   Requirement"):out.index("requirements pass") + len("requirements pass")]
details = out[:out.index("ID   Requirement")].strip()
block = "\n".join([START, "### Requirements (all checked in CI)", "", "```text", table, "```", "",
                   "### Full output", "", "```text", details, "```", END])
readme = Path("README.md").read_text()
a, b = readme.index(START), readme.index(END) + len(END)
Path("README.md").write_text(readme[:a] + block + readme[b:])
print("README.md results updated")
