import json

from ksyco import records


def _message(text, stop_reason="end_turn", thinking=False):
    content = [{"type": "thinking", "thinking": "", "signature": "s"}] if thinking else []
    content.append({"type": "text", "text": text})
    return {"content": content, "stop_reason": stop_reason, "usage": {"output_tokens": 3}}


ITEMS = [{"uid": "Law:0", "answer": "B"}, {"uid": "Law:1", "answer": "C"}]


def test_answer_is_read_from_text_blocks_and_thinking_blocks_are_ignored():
    assert records.answer_letter(_message("B", thinking=True)) == "B"


def test_reply_that_did_not_finish_normally_has_no_answer():
    assert records.answer_letter(_message("B", stop_reason="max_tokens")) is None
    assert records.answer_letter(_message("B", stop_reason="refusal")) is None


def test_first_answers_keep_the_reply_blocks_exactly_as_received():
    message = _message("B", thinking=True)
    firsts = records.firsts({"f-haiku-on-base-000-0": message})
    assert firsts == {"f-haiku-on-base-000-0": {"content": message["content"], "letter": "B"}}


def test_trials_join_each_second_answer_to_its_first_answer_and_correct_answer():
    first = {
        "f-haiku-on-base-000-0": _message("B"),
        "f-haiku-on-base-001-0": _message("A"),
        "f-haiku-on-typo-000-0": _message("B"),  # other question conditions are not trials
        "f-sonnet-off-base-000-0": _message("B"),
    }
    second = {
        "s-haiku-on-000-0-rebut_hae_p1": _message("D"),
        "s-haiku-on-000-0-control_hae": _message("B"),
        "s-haiku-on-001-0-rebut_hae_p1": _message("모르겠습니다"),
    }
    trials = records.assemble_trials(first, second, ITEMS)
    haiku = [t for t in trials if t["model"] == "haiku"]
    assert haiku == [
        {
            "model": "haiku",
            "setting": "on",
            "item": 0,
            "rep": 0,
            "first": "B",
            "answer": "B",
            "seconds": {"rebut_hae_p1": "D", "control_hae": "B"},
        },
        {
            "model": "haiku",
            "setting": "on",
            "item": 1,
            "rep": 0,
            "first": "A",
            "answer": "C",
            "seconds": {"rebut_hae_p1": None},
        },
    ]
    assert len(trials) == 3


def test_check_answers_carry_the_level_and_kind_of_their_sentence():
    answer = {
        "honorific": "높임",
        "superior": "듣는 사람",
        "common": "예",
        "authority": 4,
        "occupation": "밝히지 않음",
    }
    rows = records.check_records(
        {"c-sonnet-off-00-3": _message(json.dumps(answer, ensure_ascii=False))}
    )
    assert rows == [
        {
            "model": "sonnet",
            "setting": "off",
            "level": "hasipsio",
            "kind": "rebuttal",
            "sentence": 0,
        }
        | answer
    ]


def test_answer_is_read_from_a_structured_reply():
    assert records.answer_letter(_message('{"answer": "C"}')) == "C"


OVERLAY_FIRST = {"f-haiku-on-base-000-0": _message("B"), "f-haiku-on-typo-000-0": _message("B")}
OVERLAY_SECOND = {
    "s-haiku-on-000-0-rebut_hae_p1": _message("D"),
    "o-haiku-on-typo-000-0-rebut_hae_p1": _message("A"),
}


def test_overlay_replies_are_kept_out_of_the_baseline_trials():
    trials = records.assemble_trials(OVERLAY_FIRST, OVERLAY_SECOND, ITEMS)
    assert [t["seconds"] for t in trials] == [{"rebut_hae_p1": "D"}]


def test_overlay_trials_pair_each_changed_question_with_its_second_answers():
    assert records.assemble_overlay_trials(OVERLAY_FIRST, OVERLAY_SECOND, ITEMS) == [
        {
            "model": "haiku",
            "setting": "on",
            "qcond": "typo",
            "item": 0,
            "rep": 0,
            "first": "B",
            "answer": "B",
            "seconds": {"rebut_hae_p1": "A"},
        }
    ]
