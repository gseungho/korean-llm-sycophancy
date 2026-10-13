"""Exploratory look at the pilot replies beyond `python -m ksyco.pilot report`.

    python tools/pilot_explore.py [results/pilot]

Prints per-wording flip rates, contrasts with item-level bootstrap intervals, what happens after
a wrong first answer, and flip rates by answer stability and subject. Nothing here calls the API.
These numbers are exploratory: the contrasts were chosen after seeing the data.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ksyco import pilot, records, report
from ksyco.metrics import agreement, flip_rate

MAIN_LEVELS = ("hasipsio", "haeyo", "hae")


def wordings(level):
    return [f"rebut_{level}_p{p}" for p in (1, 2, 3)]


CONTRASTS = {
    "hasipsio - hae (main)": (wordings("hasipsio"), wordings("hae")),
    "hasipsio - haeyo (formality)": (wordings("hasipsio"), wordings("haeyo")),
    "haeyo - hae (honorific)": (wordings("haeyo"), wordings("hae")),
    "professor_haeyo - same wording no title": (["professor_haeyo"], ["rebut_haeyo_p1"]),
    "professor_hae - same wording no title": (["professor_hae"], ["rebut_hae_p1"]),
    "choseong - hae wording 1": (["choseong"], ["rebut_hae_p1"]),
    "english - mean of 3 Korean levels": (
        wordings("english"),
        [c for level in MAIN_LEVELS for c in wordings(level)],
    ),
    "hage - hae wording 1": (["rebut_hage_p1"], ["rebut_hae_p1"]),
    "haera - hae wording 1": (["rebut_haera_p1"], ["rebut_hae_p1"]),
}


def mean_rate(trials, conds):
    rates = [flip_rate(trials, cond) for cond in conds]
    rates = [rate for rate in rates if rate is not None]
    return sum(rates) / len(rates) if rates else None


def contrast(trials, conds_a, conds_b, n_boot=2000, seed=1):
    """Mean flip rate of conds_a minus conds_b with a 95% interval, resampling whole items."""
    by_item = defaultdict(list)
    for t in trials:
        by_item[t["item"]].append(t)
    ids = sorted(by_item)

    def diff(sample):
        pooled = [t for i in sample for t in by_item[i]]
        a, b = mean_rate(pooled, conds_a), mean_rate(pooled, conds_b)
        return None if a is None or b is None else a - b

    rng = random.Random(seed)
    draws = [diff(rng.choices(ids, k=len(ids))) for _ in range(n_boot)]
    draws = sorted(d for d in draws if d is not None)
    return diff(ids), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def main(out_dir):
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    check, first, second = (
        pilot.load_replies(out_dir / f"{n}.jsonl") for n in ("check", "first", "second")
    )
    groups = defaultdict(list)
    for trial in records.assemble_trials(first, second, items):
        groups[trial["model"], trial["setting"]].append(trial)

    main_conds = [c for level in MAIN_LEVELS for c in wordings(level)]
    for group, trials in sorted(groups.items()):
        wrong = [t for t in trials if t["first"] is not None and t["first"] != t["answer"]]
        n_right = sum(t["first"] == t["answer"] for t in trials)
        n_items = len({t["item"] for t in trials})
        counts = f"시행 {len(trials)}, 첫 답 맞음 {n_right}, 틀림 {len(wrong)}, 문항 {n_items}"
        print(f"\n=== {group} | {counts}")

        print("  문구별 번복률 (문구 1 / 2 / 3):")
        for level in (*MAIN_LEVELS, "english"):
            print(f"    {level:<9}", " / ".join(pct(flip_rate(trials, c)) for c in wordings(level)))

        for name, (conds_a, conds_b) in CONTRASTS.items():
            point, low, high = contrast(trials, conds_a, conds_b)
            print(f"  {name:<42} {100 * point:+5.1f}%p  [{100 * low:+5.1f}, {100 * high:+5.1f}]")

        seconds = [(t, t["seconds"][c]) for t in wrong for c in main_conds if t["seconds"].get(c)]
        if seconds:
            moved = sum(answer != t["first"] for t, answer in seconds) / len(seconds)
            fixed = sum(answer == t["answer"] for t, answer in seconds) / len(seconds)
            print(f"  첫 답이 틀렸을 때: 바꾼 비율 {pct(moved)}, 정답으로 간 비율 {pct(fixed)}")

        by_item = defaultdict(list)
        for t in trials:
            by_item[t["item"]].append(t)
        stable = [v for v in by_item.values() if agreement([t["first"] for t in v]) == 1.0]
        shaky = [v for v in by_item.values() if agreement([t["first"] for t in v]) < 1.0]
        stable_rate = pct(mean_rate(sum(stable, []), main_conds))
        shaky_rate = pct(mean_rate(sum(shaky, []), main_conds))
        print(f"  첫 답이 5번 모두 같은 문항 {len(stable)}개: {stable_rate}")
        print(f"  첫 답이 갈린 문항 {len(shaky)}개: {shaky_rate}")

        by_subject = defaultdict(list)
        for t in trials:
            by_subject[items[t["item"]]["subject"]].append(t)
        rates = {subject: pct(mean_rate(v, main_conds)) for subject, v in by_subject.items()}
        print("  과목별:", rates)

    cost = report.batch_cost(check | first | second)
    print(f"\n비용 (Batch 가격): {cost}, 합계 ${sum(cost.values()):.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/pilot"))
