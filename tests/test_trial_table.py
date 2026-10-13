from ksyco import trial_table

ITEMS = [
    {"uid": "Health:44", "question": "문항 본문 하나", "answer": "B"},
    {"uid": "Law:7", "question": "문항 본문 둘", "answer": "D"},
]


def _trial(item, rep, first, seconds, model="haiku", setting="on"):
    answer = ITEMS[item]["answer"]
    return {
        "model": model,
        "setting": setting,
        "item": item,
        "rep": rep,
        "first": first,
        "answer": answer,
        "seconds": seconds,
    }


def test_table_has_ids_and_letters_but_no_item_text(tmp_path):
    path = tmp_path / "trials.csv"
    trial_table.write([_trial(0, 0, "B", {"rebut_hae_p1": "C"})], ITEMS, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "model,setting,item,uid,rep,first,answer,rebut_hae_p1",
        "haiku,on,0,Health:44,0,B,B,C",
    ]
    assert b"\r" not in path.read_bytes()


def test_rows_come_back_as_trials_the_analysis_functions_take(tmp_path):
    path = tmp_path / "trials.csv"
    trials = [
        _trial(1, 2, "D", {"rebut_hae_p1": "D", "control_haeyo": "D"}),
        _trial(0, 0, "B", {"rebut_hae_p1": "C", "control_haeyo": "B"}),
    ]
    trial_table.write(trials, ITEMS, path)
    assert trial_table.read(path) == [
        _trial(0, 0, "B", {"control_haeyo": "B", "rebut_hae_p1": "C"}) | {"uid": "Health:44"},
        _trial(1, 2, "D", {"control_haeyo": "D", "rebut_hae_p1": "D"}) | {"uid": "Law:7"},
    ]


def test_answers_without_a_letter_and_conditions_not_run_are_left_out(tmp_path):
    path = tmp_path / "trials.csv"
    trials = [
        _trial(0, 0, None, {"rebut_hae_p1": "C"}),  # no letter in the first answer
        _trial(0, 1, "B", {"rebut_hae_p1": None, "opinion_hae_p1": "B"}),  # second answer cut off
        _trial(1, 0, "D", {"rebut_hae_p1": "A"}, model="sonnet", setting="off"),  # fewer conditions
    ]
    trial_table.write(trials, ITEMS, path)
    back = trial_table.read(path)
    assert [t["first"] for t in back] == [None, "B", "D"]
    assert back[1]["seconds"] == {"opinion_hae_p1": "B"}
    assert back[2]["seconds"] == {"rebut_hae_p1": "A"}
