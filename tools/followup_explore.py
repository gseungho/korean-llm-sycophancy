"""Extra comparisons for the follow-up experiment, chosen after seeing its report.

    python tools/followup_explore.py

Prints the flip rate of each added-phrase family and a few contrasts between families, with
item-level bootstrap intervals. Reads shared/trials.csv; nothing here calls the API. Exploratory.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from ksyco import trial_table
from ksyco.metrics import contrast, flip_rate

TRIALS = "shared/trials.csv"
FAMILIES = ("rebut", "filler", "sure", "opinion", "student", "professor", "passerby", "titleafter")
CONTRASTS = {
    "identity additions - other additions": (
        ["passerby", "titleafter", "professor", "student"],
        ["opinion", "filler", "sure"],
    ),
    "title after - title before (professor)": (["titleafter"], ["professor"]),
    "passerby - professor": (["passerby"], ["professor"]),
    "filler - opinion": (["filler"], ["opinion"]),
}


def family_conds(names, levels):
    return [f"{name}_{level}_p{p}" for name in names for level in levels for p in (1, 2, 3)]


def main():
    cfg = json.loads(Path("configs/followup1_v1.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for trial in trial_table.read(TRIALS):
        groups[trial["model"], trial["setting"]].append(trial)

    for model, setting in cfg["arms"]:
        trials, levels = groups[model, setting], cfg["levels"][model]
        print(f"\n=== {model}, {setting}")
        for family in FAMILIES:
            conds = family_conds([family], levels)
            rate = sum(flip_rate(trials, cond) for cond in conds) / len(conds)
            print(f"  {family:<11} {100 * rate:5.1f}%")
        for name, (names_a, names_b) in CONTRASTS.items():
            conds_a, conds_b = family_conds(names_a, levels), family_conds(names_b, levels)
            point, low, high = contrast(trials, conds_a, conds_b, n_boot=2000, seed=1)
            print(f"  {name:<40} {100 * point:+5.1f}%p [{100 * low:+5.1f}, {100 * high:+5.1f}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
