"""Run the pilot of docs/specs/2026-10-12-pilot.md one stage at a time.

    python -m ksyco.pilot plan      request counts, no API call
    python -m ksyco.pilot sample    download KMMLU files and draw the items
    python -m ksyco.pilot check     part 1: speech-level check
    python -m ksyco.pilot first     part 2: first answers
    python -m ksyco.pilot second    part 2: second answers (needs check and first)
    python -m ksyco.pilot report    tables from the saved replies, no API call

check, first and second each send a batch, wait for it and save the replies. A stage that was
interrupted picks up its saved batch instead of sending a new one. --no-wait sends and exits;
the batch runs on the API side with the computer off, and the same command without the flag
collects it later. --smoke runs a tiny version into results/pilot_smoke/.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ksyco import build, kmmlu, metrics, records, report
from ksyco.conditions import MODELS, OPTIONAL_LEVELS

SMOKE = {
    "items_per_subject": 1,
    "repeats": 1,
    "sonnet_thinking_on_items_per_subject": 1,
    "check_repeats": 1,
}


def load_replies(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["custom_id"]: row["message"] for row in rows}


def _write_jsonl(path, mapping, value_key):
    with open(path, "w", encoding="utf-8") as f:
        for custom_id, value in mapping.items():
            f.write(
                json.dumps({"custom_id": custom_id, value_key: value}, ensure_ascii=False) + "\n"
            )


def run_stage(name, requests, out_dir, api, chunk_size=10000, wait=True):
    """Send `requests` in batches, wait, save and return the replies; safe to rerun.

    With wait=False it only sends: the batches keep running on the API side and a later run
    of the same stage collects them.
    """
    out_path = out_dir / f"{name}.jsonl"
    if out_path.exists():
        print(f"[{name}] 이미 받은 결과가 있어 다시 보내지 않습니다: {out_path}")
        return load_replies(out_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    batch_ids = state.setdefault(name, [])
    chunks = [requests[i : i + chunk_size] for i in range(0, len(requests), chunk_size)]
    resumed = len(batch_ids)
    for chunk in chunks[resumed:]:
        batch_ids.append(api.submit(chunk))
        # Saved after every submission so an interrupted run never pays for a chunk twice.
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    sent = len(batch_ids) - resumed
    print(f"[{name}] 요청 {len(requests)}개: 새로 보낸 묶음 {sent}개, 이어받은 묶음 {resumed}개.")

    if not wait:
        print(f"[{name}] 기다리지 않고 끝냅니다. --no-wait 없이 다시 실행하면 결과를 받습니다.")
        return None

    replies, failures = {}, {}
    for batch_id in batch_ids:
        api.wait(batch_id)
        ok, bad = api.collect(batch_id)
        replies |= ok
        failures |= bad
    _write_jsonl(out_path, replies, "message")
    if failures:
        _write_jsonl(out_dir / f"{name}_failures.jsonl", failures, "result")
    print(f"[{name}] 성공 {len(replies)}개, 실패 {len(failures)}개를 저장했습니다: {out_path}")
    return replies


def plan(cfg):
    """Request counts per stage; the second stage depends on which optional levels pass."""
    n_items = len(cfg["subjects"]) * cfg["items_per_subject"]
    choices = {k: "보기" for k in "ABCD"}
    items = [
        {"uid": f"plan:{i}", "question": "문항", "choices": choices, "answer": "A"}
        for i in range(n_items)
    ]
    first = build.first_turn_requests(items, cfg)
    firsts = {r["custom_id"]: {"content": [], "letter": "A"} for r in first}
    all_pass = {model: list(OPTIONAL_LEVELS) for model in MODELS}
    return {
        "check": len(build.check_requests(cfg)),
        "first": len(first),
        "second_min": len(build.second_turn_requests(firsts, items, cfg, {})),
        "second_max": len(build.second_turn_requests(firsts, items, cfg, all_pass)),
    }


def _source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_manifest(out_dir, stage, cfg, n_requests):
    manifest = {
        "stage": stage,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "models": MODELS,
        "requests": n_requests,
        "source_sha256": _source_hash(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _sample(cfg, out_dir):
    pool = {}
    for subject in cfg["subjects"]:
        pool[subject] = kmmlu.load_subject(subject, kmmlu.download(subject, Path("data/kmmlu")))
        print(f"[sample] {subject}: 쓸 수 있는 문항 {len(pool[subject])}개")
    items = kmmlu.sample_items(pool, cfg["items_per_subject"], cfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.json"
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sample] 문항 {len(items)}개를 뽑아 저장했습니다: {path}")


def _requests_for(stage, cfg, out_dir):
    if stage == "check":
        return build.check_requests(cfg)
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    if stage == "first":
        return build.first_turn_requests(items, cfg)
    passed = metrics.passed_levels(records.check_records(load_replies(out_dir / "check.jsonl")))
    print(f"[second] 화계 확인을 통과한 화계: {passed}")
    firsts = records.firsts(load_replies(out_dir / "first.jsonl"))
    return build.second_turn_requests(firsts, items, cfg, passed)


def write_report(cfg, out_dir):
    """Build the markdown report from the saved replies of all three stages and save it."""
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    check, first, second = (
        load_replies(out_dir / f"{n}.jsonl") for n in ("check", "first", "second")
    )
    groups = defaultdict(list)
    for trial in records.assemble_trials(first, second, items):
        groups[trial["model"], trial["setting"]].append(trial)
    diffs = {}
    for group, trials in groups.items():
        if None not in (metrics.level_rate(trials, "hasipsio"), metrics.level_rate(trials, "hae")):
            diffs[group] = metrics.bootstrap_level_diff(trials, "hasipsio", "hae", seed=cfg["seed"])
    replies = check | first | second
    text = report.render(
        check=report.check_summary(records.check_records(check)),
        summaries={group: report.summarize(trials) for group, trials in sorted(groups.items())},
        diffs=diffs,
        axis=report.question_axis(first, items),
        usage=report.mean_usage(replies),
        cost=report.batch_cost(replies),
    )
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["plan", "sample", "check", "first", "second", "report"])
    parser.add_argument("--smoke", action="store_true", help="tiny run into results/pilot_smoke/")
    parser.add_argument("--no-wait", action="store_true", help="send the batch and exit")
    parser.add_argument("--config", default="configs/pilot_v1.json")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path("results/pilot")
    if args.smoke:
        cfg |= SMOKE | {"subjects": cfg["subjects"][:2]}
        out_dir = Path("results/pilot_smoke")

    if args.stage == "plan":
        for name, count in plan(cfg).items():
            print(f"{name}: {count}")
    elif args.stage == "sample":
        _sample(cfg, out_dir)
    elif args.stage == "report":
        print(write_report(cfg, out_dir))
    else:
        from ksyco.batches import BatchApi  # imported late: only these stages need the API key

        requests = _requests_for(args.stage, cfg, out_dir)
        _write_manifest(out_dir, args.stage, cfg, len(requests))
        run_stage(args.stage, requests, out_dir, BatchApi(), wait=not args.no_wait)


if __name__ == "__main__":
    main()
