import csv

from ksyco import kmmlu

COLUMNS = ["question", "answer", "A", "B", "C", "D", "Category", "Human Accuracy"]


def _row(question, answer="3", d="120"):
    return dict(zip(COLUMNS, [question, answer, "50", "90", "100", d, "Chem", "0.5"], strict=True))


def test_item_keeps_its_row_position_and_maps_the_answer_number_to_a_letter():
    item = kmmlu.to_item("Chem", 7, _row(" 물의 끓는점은? "))
    assert item == {
        "uid": "Chem:7",
        "subject": "Chem",
        "question": "물의 끓는점은?",
        "choices": {"A": "50", "B": "90", "C": "100", "D": "120"},
        "answer": "C",
    }


def test_rows_with_an_empty_choice_are_dropped_when_loading(tmp_path):
    path = tmp_path / "Chem-test.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows([_row("첫째", d=" "), _row("둘째")])
    items = kmmlu.load_subject("Chem", path)
    assert [item["uid"] for item in items] == ["Chem:1"]


def test_sampling_takes_n_per_subject_in_subject_order_and_is_reproducible():
    pool = {
        subject: [kmmlu.to_item(subject, i, _row(f"{subject} {i}")) for i in range(30)]
        for subject in ("Law", "Health")
    }
    sample = kmmlu.sample_items(pool, n=10, seed=5)
    assert [item["subject"] for item in sample] == ["Law"] * 10 + ["Health"] * 10
    assert len({item["uid"] for item in sample}) == 20
    assert sample == kmmlu.sample_items(pool, n=10, seed=5)
    assert sample != kmmlu.sample_items(pool, n=10, seed=6)


def test_items_used_before_can_be_left_out_of_the_pool():
    pool = {"Law": [kmmlu.to_item("Law", i, _row(f"q{i}")) for i in range(3)]}
    remaining = kmmlu.without(pool, {"Law:1"})
    assert [item["uid"] for item in remaining["Law"]] == ["Law:0", "Law:2"]
