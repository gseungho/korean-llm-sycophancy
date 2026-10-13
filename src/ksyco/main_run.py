"""Run the main experiment of docs/specs/2026-10-13-main.md one stage at a time.

    python -m ksyco.main_run plan      request counts, no API call
    python -m ksyco.main_run sample    draw the items (pilot items are left out)
    python -m ksyco.main_run first     first answers
    python -m ksyco.main_run second    second answers (needs first)
    python -m ksyco.main_run report    tables from the saved replies, no API call

first and second send batches and wait; --no-wait sends and exits, and the same command without
the flag collects later. An interrupted stage picks up its saved batches instead of resending.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ksyco import kmmlu, main_design, main_report, records, report
from ksyco.pilot import _write_manifest, load_replies, run_stage

OUT_DIR = Path("results/main")


def plan(cfg):
    """Request counts per stage, from the design tables (nothing is built or sent)."""
    trials = len(cfg["subjects"]) * cfg["items_per_subject"] * cfg["repeats"]
    arms = len(cfg["arms"])
    axis_arms = sum(model in cfg["question_axis_models"] for model, _ in cfg["arms"])
    overlay = len(main_design.OVERLAY_QUESTIONS) * len(main_design.OVERLAY_CONDITIONS)
    first = trials * (axis_arms * len(main_design.QUESTION_FORMS) + (arms - axis_arms))
    second = trials * (arms * len(main_design.rebuttal_conditions()) + axis_arms * overlay)
    return {"first": first, "second": second, "total": first + second}


def _sample(cfg, out_dir):
    pool = {}
    for subject in cfg["subjects"]:
        pool[subject] = kmmlu.load_subject(subject, kmmlu.download(subject, Path("data/kmmlu")))
    used = json.loads(Path(cfg["exclude_items"]).read_text(encoding="utf-8"))
    pool = kmmlu.without(pool, {item["uid"] for item in used})
    items = kmmlu.sample_items(pool, cfg["items_per_subject"], cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.json"
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sample] 앞서 쓴 문항 {len(used)}개를 빼고 {len(items)}개를 뽑아 저장했습니다: {path}")


def write_report(cfg, out_dir):
    """Build the markdown report from the saved replies and save it."""
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    first, second = (load_replies(out_dir / f"{name}.jsonl") for name in ("first", "second"))
    groups = defaultdict(list)
    for trial in records.assemble_trials(first, second, items):
        groups[trial["model"], trial["setting"]].append(trial)
    results = {
        tuple(arm): main_report.analyze(groups[tuple(arm)], cfg["n_boot"], cfg["seed"])
        for arm in cfg["arms"]
    }
    replies = first | second
    text = main_report.render(
        results,
        axis=report.question_axis(first, items),
        overlay=main_report.overlay_summary(records.assemble_overlay_trials(first, second, items)),
        usage=report.mean_usage(replies),
        cost=report.batch_cost(replies),
    )
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["plan", "sample", "first", "second", "report"])
    parser.add_argument("--no-wait", action="store_true", help="send the batches and exit")
    parser.add_argument("--config", default="configs/main_v1.json")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.stage == "plan":
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
            requests = main_design.second_requests(firsts, items, cfg)
        _write_manifest(OUT_DIR, args.stage, cfg, len(requests))
        run_stage(args.stage, requests, OUT_DIR, BatchApi(), wait=not args.no_wait)


if __name__ == "__main__":
    main()
