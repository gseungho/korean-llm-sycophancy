import json

from ksyco import main_design, main_report, main_run

TINY = {
    "seed": 1,
    "subjects": ["A", "B"],
    "items_per_subject": 3,
    "repeats": 2,
    "arms": [["haiku", "on"], ["haiku", "off"], ["sonnet", "off"]],
    "question_axis_models": ["haiku"],
    "typo_rate": 0.2,
    "max_tokens": 4096,
    "n_boot": 200,
}


def _items(n):
    choices = {"A": "가", "B": "나", "C": "다", "D": "라"}
    return [
        {"uid": f"S:{i}", "question": f"질문 {i} 은 무엇인가", "choices": choices, "answer": "B"}
        for i in range(n)
    ]


def test_plan_counts_match_the_spec():
    cfg = json.loads(open("configs/main_v1.json", encoding="utf-8").read())
    assert main_run.plan(cfg) == {
        "first": 15000 + 15000 + 1500,
        "second": 81000 + 81000 + 63000,
        "total": 256500,
    }


def test_plan_counts_agree_with_the_requests_that_are_actually_built():
    items = _items(6)
    first = main_design.first_requests(items, TINY)
    content = [{"type": "text", "text": "B"}]
    firsts = {r["custom_id"]: {"content": content, "letter": "B"} for r in first}
    second = main_design.second_requests(firsts, items, TINY)
    assert main_run.plan(TINY) == {
        "first": len(first),
        "second": len(second),
        "total": len(first) + len(second),
    }


def _trial(item, flips):
    """A first-correct trial whose second answer changes exactly under the conditions in `flips`."""
    seconds = {cond: ("C" if flips(cond) else "B") for cond in main_design.rebuttal_conditions()}
    return {
        "model": "haiku",
        "setting": "on",
        "item": item,
        "rep": 0,
        "first": "B",
        "answer": "B",
        "seconds": seconds,
    }


def _gives_way_to_formal_and_untitled(cond):
    # Changes its answer under any untitled rebuttal except plain speech, never under a title.
    return cond.startswith("rebut_") and not cond.startswith("rebut_hae_")


TRIALS = [_trial(i, _gives_way_to_formal_and_untitled) for i in range(20)]


def test_primary_comparisons_get_a_verdict():
    primary = main_report.analyze(TRIALS, n_boot=200, seed=0)["primary"]
    assert primary["speech_level"]["point"] == 1.0  # formal always flips, plain never
    assert primary["speech_level"]["verdict"] == "supported"
    assert primary["professor"]["point"] == -0.5  # title never flips; untitled haeyo yes, hae no
    assert primary["professor"]["verdict"] == "supported"


def test_exploratory_comparisons_are_reported_without_a_verdict():
    exploratory = main_report.analyze(TRIALS, n_boot=200, seed=0)["exploratory"]
    assert exploratory["professor_minus_student"]["point"] == 0.0
    assert all("verdict" not in result for result in exploratory.values())


def test_report_text_states_each_verdict_in_korean():
    results = {("haiku", "on"): main_report.analyze(TRIALS, n_boot=200, seed=0)}
    text = main_report.render(results, axis={}, overlay={}, usage={}, cost={})
    assert "예측이 맞음" in text
    assert "하십시오체 빼기 해체" in text


def test_report_is_built_end_to_end_from_saved_replies(tmp_path):
    """Requests are built, answered by a stand-in for the API, saved, and turned into a report."""
    items = _items(6)
    usage = {"input_tokens": 300, "output_tokens": 5}

    def save(name, replies):
        lines = [json.dumps({"custom_id": k, "message": v}) for k, v in replies.items()]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    def reply(text):
        return {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": usage,
        }

    first = {r["custom_id"]: reply("B") for r in main_design.first_requests(items, TINY)}
    from ksyco import records

    second_requests = main_design.second_requests(records.firsts(first), items, TINY)
    # The stand-in model gives way to every formal-level rebuttal and to nothing else.
    second = {
        r["custom_id"]: reply(
            '{"answer":"C"}' if "hasipsio" in r["custom_id"] else '{"answer":"B"}'
        )
        for r in second_requests
    }
    (tmp_path / "items.json").write_text(json.dumps(items), encoding="utf-8")
    save("first", first)
    save("second", second)

    text = main_run.write_report(TINY, tmp_path)

    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text
    assert text.count("예측이 맞음") == 3  # the speech-level comparison, in each of the three arms
    assert "## 6. 질문을 바꾼 뒤의 번복률" in text
