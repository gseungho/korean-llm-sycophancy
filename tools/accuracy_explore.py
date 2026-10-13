"""Accuracy before and after the second turn, chosen after the results were seen.

    python tools/accuracy_explore.py

Prints, for each model setting, how often the answer was correct before and after a group of
second-turn conditions, and how often a wrong first answer became the correct one. Reads
shared/trials.csv; nothing here calls the API. Exploratory.
"""

import sys
from collections import defaultdict

from ksyco import trial_table
from ksyco.metrics import accuracy_before_after

TRIALS = "shared/trials.csv"
ARMS = (("haiku", "on"), ("haiku", "off"), ("sonnet", "off"))
# Polite-style (haeyo) conditions, which every model setting has.
GROUPS = {
    "re-ask without rebuttal": ["control_haeyo"],
    "bare rebuttal": [f"rebut_haeyo_p{p}" for p in (1, 2, 3)],
    "filler + rebuttal": [f"filler_haeyo_p{p}" for p in (1, 2, 3)],
    "opinion + rebuttal": [f"opinion_haeyo_p{p}" for p in (1, 2, 3)],
    "professor + rebuttal": [f"professor_haeyo_p{p}" for p in (1, 2, 3)],
    "rebuttal + professor": [f"titleafter_haeyo_p{p}" for p in (1, 2, 3)],
}


def mean(values):
    return sum(values) / len(values)


def corrected(trials, conds):
    """Among wrong first answers, the share of second answers that are correct."""
    hits = [
        t["seconds"][cond] == t["answer"]
        for t in trials
        for cond in conds
        if t["first"] not in (None, t["answer"]) and t["seconds"].get(cond) is not None
    ]
    return mean(hits)


def main():
    groups = defaultdict(list)
    for trial in trial_table.read(TRIALS):
        groups[trial["model"], trial["setting"]].append(trial)

    for arm in ARMS:
        trials = groups[arm]
        first = mean([t["first"] == t["answer"] for t in trials])
        print(
            f"\n=== {arm[0]}, {arm[1]}: first answers correct {100 * first:.1f}% of {len(trials)}"
        )
        print(f"  {'second turn':<26}{'before':>8}{'after':>8}{'change':>9}{'wrong -> right':>16}")
        for name, conds in GROUPS.items():
            pairs = [accuracy_before_after(trials, cond) for cond in conds]
            before, after = (100 * mean([pair[i] for pair in pairs]) for i in (0, 1))
            fixed = 100 * corrected(trials, conds)
            print(f"  {name:<26}{before:7.1f}%{after:7.1f}%{after - before:+8.1f}{fixed:15.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
