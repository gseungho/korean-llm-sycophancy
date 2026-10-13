"""How much the two pre-registered effects depend on the rebuttal wording.

    python tools/wording_explore.py

Written after an outside review of the repository pointed out that the wordings were treated as
fixed. Prints each effect per wording with an item-level bootstrap interval, then an interval
that treats the wordings as the sampled unit (a t interval over the three per-wording effects), and
where Sonnet's first answers without a letter came from. Reads shared/trials.csv; nothing here
calls the API. Exploratory.
"""

import sys
from collections import Counter, defaultdict
from statistics import mean, stdev

from scipy.stats import t as t_dist

from ksyco import trial_table
from ksyco.metrics import contrast

TRIALS = "shared/trials.csv"
ARMS = (("haiku", "on"), ("haiku", "off"), ("sonnet", "off"))
WORDINGS = (1, 2, 3)


def over_sentences(effects):
    """Mean of per-wording effects with a 95% t interval, the wordings being the sample."""
    half = t_dist.ppf(0.975, len(effects) - 1) * stdev(effects) / len(effects) ** 0.5
    return mean(effects), mean(effects) - half, mean(effects) + half


def show(trials, name, pairs):
    """`pairs` maps a wording to (its conditions, the conditions they are compared with)."""
    print(f"  -- {name}")
    effects = []
    for label, (conds, references) in pairs.items():
        point, low, high = (100 * v for v in contrast(trials, conds, references, seed=1))
        effects.append(point)
        print(f"  wording {label:<14}{point:+6.1f}%p [{low:+6.1f}, {high:+6.1f}]")
    point, low, high = over_sentences(effects)
    print(f"  {'over wordings (t)':<22}{point:+6.1f}%p [{low:+6.1f}, {high:+6.1f}]")


def main():
    groups = defaultdict(list)
    for trial in trial_table.read(TRIALS):
        groups[trial["model"], trial["setting"]].append(trial)

    for arm in ARMS:
        trials = groups[arm]
        print(f"\n=== {arm[0]}, {arm[1]}")
        speech = {p: ([f"rebut_hasipsio_p{p}"], [f"rebut_hae_p{p}"]) for p in WORDINGS}
        show(trials, "hasipsio - hae", speech)
        title = {
            p: (
                [f"professor_haeyo_p{p}", f"professor_hae_p{p}"],
                [f"rebut_haeyo_p{p}", f"rebut_hae_p{p}"],
            )
            for p in WORDINGS
        }
        show(trials, "professor title - no title (two speech levels)", title)

    sonnet = groups["sonnet", "off"]
    missing = Counter(t["uid"] for t in sonnet if t["first"] is None)
    print(f"\n=== sonnet, off: {sum(missing.values())} first answers without a letter")
    print(f"  items: {len(missing)}, by subject: {dict(Counter(u.split(':')[0] for u in missing))}")
    print(f"  items with all five repeats missing: {sum(n == 5 for n in missing.values())}")
    for name, arm in (("sonnet, off", ("sonnet", "off")), ("haiku, on", ("haiku", "on"))):
        answered = [t for t in groups[arm] if t["first"] is not None]
        inside = [t["first"] == t["answer"] for t in answered if t["uid"] in missing]
        outside = [t["first"] == t["answer"] for t in answered if t["uid"] not in missing]
        print(
            f"  first-answer accuracy of {name}: on these items {100 * mean(inside):.1f}% "
            f"(n={len(inside)}), on the other items {100 * mean(outside):.1f}%"
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
