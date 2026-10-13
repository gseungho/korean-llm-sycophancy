"""Build shared/, the part of the results that can be published, from results/.

    python tools/share.py

Writes shared/trials.csv (item ids and answer letters of the main and follow-up experiments, no
item text) and copies the three aggregate reports. Needs the saved replies in results/, which
are not in the repository. Nothing here calls the API.
"""

import json
import shutil
import sys
from pathlib import Path

from ksyco import pilot, records, trial_table

OUT_DIR = Path("shared")
REPORTS = {"pilot": "results/pilot", "main": "results/main", "followup1": "results/followup1"}


def main():
    source = Path("results/main")
    items = json.loads((source / "items.json").read_text(encoding="utf-8"))
    first = pilot.load_replies(source / "first.jsonl")
    second = pilot.load_replies(source / "second.jsonl")
    second |= pilot.load_replies("results/followup1/second.jsonl")
    trials = records.assemble_trials(first, second, items)

    (OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    trial_table.write(trials, items, OUT_DIR / "trials.csv")
    print(f"{len(trials)} trials -> {OUT_DIR / 'trials.csv'}")
    for name, folder in REPORTS.items():
        shutil.copyfile(Path(folder) / "report.md", OUT_DIR / "reports" / f"{name}.md")
        print(f"copied {folder}/report.md -> {OUT_DIR / 'reports' / (name + '.md')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
