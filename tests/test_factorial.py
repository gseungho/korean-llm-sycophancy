import pytest

from ksyco import factorial


def _trial(item, first, seconds, answer="B"):
    return {"item": item, "first": first, "answer": answer, "seconds": seconds}


def test_item_table_gives_the_flip_share_of_first_correct_trials_per_cell():
    cells = {("lo",): "rebut_hae_p1", ("hi",): "rebut_hasipsio_p1"}
    trials = [
        _trial(0, "B", {"rebut_hae_p1": "C", "rebut_hasipsio_p1": "B"}),
        _trial(0, "B", {"rebut_hae_p1": "B", "rebut_hasipsio_p1": "B"}),
        _trial(0, "A", {"rebut_hae_p1": "B", "rebut_hasipsio_p1": "B"}),  # first answer was wrong
        _trial(1, "B", {"rebut_hae_p1": "D", "rebut_hasipsio_p1": "D"}),
    ]
    assert factorial.item_table(trials, cells) == {
        0: {("lo",): 0.5, ("hi",): 0.0},
        1: {("lo",): 1.0, ("hi",): 1.0},
    }


def test_item_table_drops_items_with_an_empty_cell():
    cells = {("lo",): "rebut_hae_p1", ("hi",): "rebut_hasipsio_p1"}
    trials = [
        _trial(0, "B", {"rebut_hae_p1": "C", "rebut_hasipsio_p1": None}),  # no second answer
        _trial(1, "A", {"rebut_hae_p1": "B", "rebut_hasipsio_p1": "B"}),  # never correct
        _trial(2, "B", {"rebut_hae_p1": "B", "rebut_hasipsio_p1": "C"}),
    ]
    assert list(factorial.item_table(trials, cells)) == [2]


def _rows(table, factors):
    return {row["source"]: row for row in factorial.anova(table, factors)}


def test_anova_of_a_two_by_two_in_two_blocks_matches_the_hand_calculation():
    # Treatment totals (1)=3, a=6, b=6, ab=11: contrasts 8, 8 and 2, each squared over 8 runs.
    table = {
        "block1": {("lo", "lo"): 1, ("hi", "lo"): 3, ("lo", "hi"): 2, ("hi", "hi"): 6},
        "block2": {("lo", "lo"): 2, ("hi", "lo"): 3, ("lo", "hi"): 4, ("hi", "hi"): 5},
    }
    rows = _rows(table, {"A": ["lo", "hi"], "B": ["lo", "hi"]})
    assert list(rows) == ["block", "A", "B", "A x B", "error", "total"]
    assert rows["block"]["ss"] == pytest.approx(0.5)
    assert rows["A"]["ss"] == pytest.approx(8.0)
    assert rows["B"]["ss"] == pytest.approx(8.0)
    assert rows["A x B"]["ss"] == pytest.approx(0.5)
    assert rows["error"]["ss"] == pytest.approx(2.5)
    assert rows["total"]["ss"] == pytest.approx(19.5)
    assert [rows[s]["df"] for s in rows] == [1, 1, 1, 1, 3, 7]
    assert rows["A"]["f"] == pytest.approx(9.6)
    assert rows["A x B"]["f"] == pytest.approx(0.6)
    assert 0.05 < rows["A"]["p"] < 0.06  # F(1, 3) reaches 0.05 at 10.13
    assert rows["error"]["f"] is None and rows["block"]["p"] is None


def test_anova_handles_a_factor_with_three_levels():
    table = {
        "block1": dict(zip(_cells(), [0, 1, 2, 1, 2, 3], strict=True)),
        "block2": dict(zip(_cells(), [1, 2, 3, 2, 3, 5], strict=True)),
    }
    rows = _rows(table, {"A": ["lo", "hi"], "C": [1, 2, 3]})
    assert rows["C"]["ss"] == pytest.approx(249 / 4 - 625 / 12)
    assert rows["C"]["df"] == 2
    assert rows["A"]["ss"] == pytest.approx(337 / 6 - 625 / 12)
    assert rows["A x C"]["ss"] == pytest.approx(1 / 6)
    assert rows["A x C"]["df"] == 2
    assert rows["error"]["ss"] == pytest.approx(5 / 12)
    assert rows["error"]["df"] == 5
    assert rows["C"]["f"] == pytest.approx((249 / 4 - 625 / 12) / 2 / (1 / 12))


def _cells():
    return [(a, c) for a in ("lo", "hi") for c in (1, 2, 3)]


def test_anova_p_value_is_the_upper_tail_of_the_f_distribution():
    # Any 2 x 2 in 4 blocks has 9 error degrees of freedom; F(1, 9) reaches 0.05 at 5.117.
    table = {
        b: {("lo", "lo"): 0 + e, ("hi", "lo"): 2 - e, ("lo", "hi"): 0 - e, ("hi", "hi"): 2 + e}
        for b, e in zip("wxyz", (1, -1, 1, -1), strict=True)
    }
    row = _rows(table, {"A": ["lo", "hi"], "B": ["lo", "hi"]})["A"]
    from scipy.stats import f as f_dist

    assert row["p"] == pytest.approx(f_dist.sf(row["f"], 1, 9))
    assert f_dist.sf(5.117, 1, 9) == pytest.approx(0.05, abs=1e-4)
