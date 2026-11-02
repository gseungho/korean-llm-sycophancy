"""The verdicts of the confirmation on new items, from the shared table.

    python tools/confirm_report.py

Prints the same tables as `python -m ksyco.confirm report` without the saved replies. Nothing
here calls the API.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from ksyco import confirm, trial_table


def main():
    cfg = json.loads(Path("configs/confirm1_v1.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for trial in trial_table.read("shared/confirm1_trials.csv"):
        groups[trial["model"], trial["setting"]].append(trial)
    results = {
        tuple(arm): confirm.analyze(groups[tuple(arm)], cfg["n_boot"], cfg["seed"])
        for arm in cfg["arms"]
    }
    print(confirm.render(results))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
