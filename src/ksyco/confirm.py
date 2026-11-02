"""New items for the predicate-by-speech-level contrast (docs/specs/2026-11-02-confirmation.md).

    python -m ksyco.confirm sentences  the 36 second-turn sentences, no API call
    python -m ksyco.confirm plan       request counts, no API call
    python -m ksyco.confirm sample     draw the items (pilot and main items are left out)
    python -m ksyco.confirm first      first answers
    python -m ksyco.confirm second     second answers (needs first)
    python -m ksyco.confirm report     tables from the saved replies, no API call

first and second send batches and wait; --no-wait sends and exits, and the same command without
the flag collects later. An interrupted stage picks up its saved batches instead of resending.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ksyco import kmmlu, main_design, records, report, wordings
from ksyco.build import ANSWER_FORMAT, base_params, format_question, parse_id
from ksyco.main_report import SETTING_NAMES, VERDICT_NAMES
from ksyco.metrics import verdict
from ksyco.pilot import _write_manifest, load_replies, run_stage

OUT_DIR = Path("results/confirm1")
# The two predicates compared, as indices into wordings.PREDICATES.
WRONG, NOT_ANSWER = 0, 2
FRAMES = [frame for frame in wordings.frames() if frame[2] in (WRONG, NOT_ANSWER)]


def conditions():
    """Condition name -> second-turn sentence."""
    return {
        wordings.condition_name(frame, level): wordings.sentence(frame, level)
        for frame in FRAMES
        for level in wordings.LEVELS
    }


def plan(cfg):
    """Request counts per stage, assuming every first answer yields a letter."""
    first = len(cfg["subjects"]) * cfg["items_per_subject"] * cfg["repeats"] * len(cfg["arms"])
    second = first * len(conditions())
    return {"first": first, "second": second, "total": first + second}


def draw_items(pool, used_uids, n, seed):
    """`n` items per subject that no earlier experiment used; fails if a subject has too few."""
    return kmmlu.sample_items(kmmlu.without(pool, used_uids), n, seed)


def second_requests(firsts, items, cfg):
    """Every condition after each usable first answer, arm by arm in the order of the config."""
    requests = []
    for model, setting in cfg["arms"]:
        for first_id, first in firsts.items():
            key = parse_id(first_id)
            if (key["model"], key["setting"]) != (model, setting) or first["letter"] is None:
                continue
            i, rep = key["item"], key["rep"]
            history = [
                {"role": "user", "content": format_question(items[i])},
                {"role": "assistant", "content": first["content"]},
            ]
            for cond, text in conditions().items():
                params = base_params(model, setting, cfg["max_tokens"]) | {
                    "messages": history + [{"role": "user", "content": text}],
                    "output_config": ANSWER_FORMAT,
                }
                custom_id = f"s-{model}-{setting}-{i:03d}-{rep}-{cond}"
                requests.append({"custom_id": custom_id, "params": params})
    return requests


def analyze(trials, n_boot=10000, seed=0):
    """The pre-registered contrast of one arm, its verdict, and the value of each frame."""
    contrast = wordings.predicate_contrast(
        trials, FRAMES, WRONG, NOT_ANSWER, level=0.99, n_boot=n_boot, seed=seed
    )
    point, low, high = contrast["point"], contrast["low"], contrast["high"]
    return {
        "items": contrast["items"],
        "correct": sum(t["first"] == t["answer"] for t in trials),
        "wrong": contrast["first"],
        "not_answer": contrast["second"],
        "point": point,
        "low": low,
        "high": high,
        "verdict": verdict(point, low, high, +1),
        "effects": wordings.analyze(trials, FRAMES, n_boot=1)["effects"],
    }


def _pp(value):
    return f"{100 * value:+.1f}%p"


def render(results):
    """The report as markdown. `results` maps (model, setting) to the output of analyze."""
    out = [
        "# 새 문항으로 확인한 서술어와 화계의 교호작용",
        "",
        "화계의 효과는 하십시오체 번복률에서 해체 번복률을 뺀 값이다. "
        '대비는 "틀렸습니다"의 효과에서 "정답이 아닙니다"의 효과를 뺀 값이다. '
        "판정 규칙은 설계서 5절에 있다.",
        "",
        "## 1. 미리 정한 비교 (99% 구간)",
        "",
        '| 모델과 설정 | 문항 | 첫 답을 맞힌 문항 | "틀렸습니다" | "정답이 아닙니다" '
        "| 대비 | 99% 구간 | 판정 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (model, setting), r in results.items():
        out.append(
            f"| {model}, {SETTING_NAMES[setting]} | {r['items']} | {r['correct']} "
            f"| {_pp(r['wrong'])} | {_pp(r['not_answer'])} | {_pp(r['point'])} "
            f"| {_pp(r['low'])} ~ {_pp(r['high'])} | {VERDICT_NAMES[r['verdict']]} |"
        )
    out += ["", "## 2. 문구 틀별 화계의 효과 (판정 없음)", ""]
    for (model, setting), r in results.items():
        out += [f"### {model}, {SETTING_NAMES[setting]}", ""]
        out += ["| 문구 (하십시오체로 쓴 꼴) | 화계의 효과 |", "| --- | --- |"]
        for frame in FRAMES:
            sentence = wordings.sentence(frame, "hasipsio")
            out.append(f"| {sentence} | {_pp(r['effects'][frame])} |")
        out.append("")
    return "\n".join(out)


def write_report(cfg, out_dir):
    """Build the markdown report from the saved replies and save it."""
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    first, second = (load_replies(out_dir / f"{name}.jsonl") for name in ("first", "second"))
    groups = defaultdict(list)
    for trial in records.assemble_trials(first, second, items):
        groups[trial["model"], trial["setting"]].append(trial)
    results = {
        tuple(arm): analyze(groups[tuple(arm)], cfg["n_boot"], cfg["seed"]) for arm in cfg["arms"]
    }
    text = render(results)
    cost = report.batch_cost(first | second)
    text += "\n## 3. 비용 (Batch 가격)\n\n" + "\n".join(
        f"- {model}: ${dollars:.2f}" for model, dollars in cost.items()
    )
    (out_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def _sample(cfg, out_dir):
    pool = {}
    for subject in cfg["subjects"]:
        pool[subject] = kmmlu.load_subject(subject, kmmlu.download(subject, Path("data/kmmlu")))
    used = set()
    for path in cfg["exclude_items"]:
        used |= {item["uid"] for item in json.loads(Path(path).read_text(encoding="utf-8"))}
    items = draw_items(pool, used, cfg["items_per_subject"], cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.json"
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sample] 앞서 쓴 문항 {len(used)}개를 빼고 {len(items)}개를 뽑아 저장했습니다: {path}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    stages = ["sentences", "plan", "sample", "first", "second", "report"]
    parser.add_argument("stage", choices=stages)
    parser.add_argument("--no-wait", action="store_true", help="send the batches and exit")
    parser.add_argument("--config", default="configs/confirm1_v1.json")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.stage == "sentences":
        for frame in FRAMES:
            print(f"{wordings.sentence(frame, 'hasipsio')}  /  {wordings.sentence(frame, 'hae')}")
    elif args.stage == "plan":
        for name, count in plan(cfg).items():
            print(f"{name}: {count}")
    elif args.stage == "sample":
        _sample(cfg, OUT_DIR)
    elif args.stage == "report":
        print(write_report(cfg, OUT_DIR))
    else:
        from ksyco.batches import BatchApi  # imported late: only these stages need the API key

        items = json.loads((OUT_DIR / "items.json").read_text(encoding="utf-8"))
        if args.stage == "first":
            requests = main_design.first_requests(items, cfg)
        else:
            firsts = records.firsts(load_replies(OUT_DIR / "first.jsonl"))
            requests = second_requests(firsts, items, cfg)
        _write_manifest(OUT_DIR, args.stage, cfg, len(requests))
        run_stage(args.stage, requests, OUT_DIR, BatchApi(), wait=not args.no_wait)


if __name__ == "__main__":
    main()
