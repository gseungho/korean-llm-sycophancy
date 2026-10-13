"""A closer look at the wording experiment, chosen after its report was seen.

    python tools/wordings_explore.py

Reproduces the pre-registered comparison from shared/trials.csv, then prints intervals for two
readings of what the 54 frames stand for (after a second outside review), the speech-level
difference for the frames sharing each opener, subject and predicate with a two-way bootstrap
interval, how well the per-frame differences of two model settings agree, and the frames with
the largest differences. Nothing here calls the API. Everything after the first table is
exploratory.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

from ksyco import trial_table, wordings

TRIALS = "shared/trials.csv"


def main():
    cfg = json.loads(Path("configs/wordings1_v1.json").read_text(encoding="utf-8"))
    table = trial_table.read(TRIALS)
    # The item subset is drawn by subject, and the table carries the subject in the uid.
    subjects = {t["item"]: t["uid"].split(":")[0] for t in table}
    items = [{"subject": subjects[i]} for i in range(len(subjects))]
    groups = defaultdict(list)
    for trial in table:
        groups[trial["model"], trial["setting"]].append(trial)

    effects = {}
    for model, setting in cfg["arms"]:
        chosen = wordings.arm_items(items, model, cfg)
        trials = [
            t for t in groups[model, setting] if t["rep"] == cfg["rep"] and t["item"] in chosen
        ]
        new = wordings.new_frames()
        result = wordings.analyze(trials, new, n_boot=cfg["n_boot"], seed=cfg["seed"])
        effects[model, setting] = result["effects"]
        print(f"\n=== {model}, {setting}")
        print(
            f"  pre-registered: {100 * result['point']:+.1f}%p "
            f"[{100 * result['low']:+.1f}, {100 * result['high']:+.1f}] {result['verdict']}, "
            f"sd over frames {100 * result['sd']:.1f}%p"
        )
        # What the 54 frames stand for decides the interval: the frames themselves (only items
        # resampled) or other wordings built the same way (their components resampled too).
        for label, components in (("items only", False), ("items and components", True)):
            r = wordings.grid_interval(
                trials, components=components, n_boot=cfg["n_boot"], seed=cfg["seed"]
            )
            print(
                f"  all 54 frames, resampling {label}: {100 * r['point']:+.1f}%p "
                f"[{100 * r['low']:+.1f}, {100 * r['high']:+.1f}]"
            )
        by_predicate = wordings.analyze(trials, wordings.frames(), n_boot=1)["by_component"]
        means = np.array(by_predicate["predicate"])
        half = t_dist.ppf(0.995, len(means) - 1) * means.std(ddof=1) / len(means) ** 0.5
        print(
            f"  t interval over the 6 predicate means: {100 * means.mean():+.1f}%p "
            f"[{100 * (means.mean() - half):+.1f}, {100 * (means.mean() + half):+.1f}]"
        )
        for axis, (name, entries) in enumerate(wordings.COMPONENTS.items()):
            for index, entry in enumerate(entries):
                shown = entry if isinstance(entry, str) else entry["hasipsio"].strip() or "(none)"
                part = [frame for frame in new if frame[axis] == index]
                r = wordings.analyze(trials, part, level=0.95, n_boot=2000, seed=cfg["seed"])
                print(
                    f"  {name:<10}{shown:<12}{100 * r['point']:+6.1f}%p "
                    f"[{100 * r['low']:+6.1f}, {100 * r['high']:+6.1f}]  frames {len(part)}"
                )
        ranked = sorted(result["effects"].items(), key=lambda pair: pair[1])
        for frame, d in ranked[:3] + ranked[-3:]:
            print(f"  {100 * d:+6.1f}%p  {wordings.sentence(frame, 'hasipsio')}")

    print("\n=== agreement of the per-frame differences between settings (correlation)")
    arms = [tuple(arm) for arm in cfg["arms"]]
    for a, first in enumerate(arms):
        for second in arms[a + 1 :]:
            frames = wordings.new_frames()
            x = [effects[first][frame] for frame in frames]
            y = [effects[second][frame] for frame in frames]
            print(f"  {first} vs {second}: {np.corrcoef(x, y)[0, 1]:+.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
