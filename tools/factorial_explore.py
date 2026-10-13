"""The saved answers read as factorial designs, chosen after the results were seen.

    python tools/factorial_explore.py

Prints, for each model setting, the 2 x 2 of formality and honorific with its three effects, and
analysis-of-variance tables with items as blocks. Reads shared/trials.csv; nothing here calls
the API. Exploratory.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ksyco import trial_table
from ksyco.factorial import anova, item_table
from ksyco.metrics import contrast, flip_rate

TRIALS = "shared/trials.csv"
ARMS = (("haiku", "on"), ("haiku", "off"), ("sonnet", "off"))
# The four speech levels as a 2 x 2: (formality, honorific) -> level.
SPEECH = {
    ("formal", "honorific"): "hasipsio",
    ("formal", "plain"): "haera",
    ("informal", "honorific"): "haeyo",
    ("informal", "plain"): "hae",
}
SPEECH_EFFECTS = {
    "formality": (("hasipsio", "haera"), ("haeyo", "hae")),
    "honorific": (("hasipsio", "haeyo"), ("haera", "hae")),
    "interaction (half the difference of differences)": (("hasipsio", "hae"), ("haera", "haeyo")),
}
PHRASES = ("rebut", "filler", "sure", "opinion", "student", "professor", "passerby", "titleafter")
WORDINGS = (1, 2, 3)


def conds(families, levels):
    return [f"{family}_{level}_p{p}" for family in families for level in levels for p in WORDINGS]


def mean_rate(trials, names):
    return sum(flip_rate(trials, cond) for cond in names) / len(names)


def print_anova(trials, cells, factors):
    table = item_table(trials, cells)
    means = [sum(row.values()) / len(row) for row in table.values()]
    print(
        f"  items kept as blocks: {len(table)} "
        f"(flipped in no trial: {means.count(0)}, in every trial: {means.count(1)})"
    )
    print(f"  {'source':<32}{'SS':>10}{'df':>7}{'MS':>10}{'F':>9}{'p':>9}{'share':>8}")
    rows = anova(table, factors)
    total = rows[-1]["ss"]
    for r in rows:
        f = "" if r["f"] is None else f"{r['f']:.2f}"
        p = "" if r["p"] is None else f"{r['p']:.4f}"
        share = f"{r['ss'] / total:.2%}"
        print(
            f"  {r['source']:<32}{r['ss']:10.3f}{r['df']:7d}{r['ms']:10.4f}{f:>9}{p:>9}{share:>8}"
        )


def main():
    cfg = json.loads(Path("configs/followup1_v1.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for trial in trial_table.read(TRIALS):
        groups[trial["model"], trial["setting"]].append(trial)

    for model, setting in ARMS:
        trials, levels = groups[model, setting], cfg["levels"][model]
        correct = Counter(t["item"] for t in trials if t["first"] == t["answer"])
        print(f"\n=== {model}, {setting}")
        print(
            f"  first-correct repeats per item: {dict(sorted(Counter(correct.values()).items()))}"
        )

        print("  -- formality x honorific (flip rate of each level, mean of three wordings)")
        for (formality, honorific), level in SPEECH.items():
            rate = mean_rate(trials, conds(["rebut"], [level]))
            print(f"  {formality:<9}{honorific:<10}{level:<9}{100 * rate:5.1f}%")
        for name, (plus, minus) in SPEECH_EFFECTS.items():
            point, low, high = contrast(
                trials, conds(["rebut"], plus), conds(["rebut"], minus), n_boot=2000, seed=1
            )
            print(f"  {name:<50}{100 * point:+5.1f}%p [{100 * low:+5.1f}, {100 * high:+5.1f}]")
        cells = {
            (*key, p): f"rebut_{level}_p{p}" for key, level in SPEECH.items() for p in WORDINGS
        }
        factors = {
            "formality": ["formal", "informal"],
            "honorific": ["honorific", "plain"],
            "wording": list(WORDINGS),
        }
        print_anova(trials, cells, factors)

        print("  -- added phrase x speech level x wording")
        for phrase in PHRASES:
            rates = [100 * mean_rate(trials, conds([phrase], [level])) for level in levels]
            line = "  ".join(
                f"{level} {rate:5.1f}%" for level, rate in zip(levels, rates, strict=True)
            )
            if len(levels) == 2:
                point, low, high = contrast(
                    trials,
                    conds([phrase], ["haeyo"]),
                    conds([phrase], ["hae"]),
                    n_boot=2000,
                    seed=1,
                )
                line += (
                    f"  haeyo - hae {100 * point:+5.1f}%p [{100 * low:+5.1f}, {100 * high:+5.1f}]"
                )
            print(f"  {phrase:<11}{line}")
        cells = {
            (phrase, level, p): f"{phrase}_{level}_p{p}"
            for phrase in PHRASES
            for level in levels
            for p in WORDINGS
        }
        factors = {"phrase": list(PHRASES), "level": list(levels), "wording": list(WORDINGS)}
        if len(levels) == 1:
            cells = {(phrase, p): cond for (phrase, _, p), cond in cells.items()}
            del factors["level"]
        print_anova(trials, cells, factors)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
