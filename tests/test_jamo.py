import random

from ksyco.jamo import add_typos, neighbors, substitute_jamo


def _parts(syllable):
    code = ord(syllable) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def test_vowel_neighbors_are_the_adjacent_vowel_keys():
    assert neighbors("ㅡ") == ["ㅜ"]


def test_consonant_next_to_a_vowel_key_only_gets_consonant_neighbors():
    # On the keyboard ㅅ sits between ㄱ and ㅛ; swapping in a vowel would break the syllable.
    assert neighbors("ㅅ") == ["ㄱ"]


def test_substitution_changes_exactly_one_of_initial_or_vowel_and_keeps_the_final():
    rng = random.Random(0)
    for syllable in "은것답문제":
        out = substitute_jamo(syllable, rng)
        cho, jung, jong = _parts(syllable)
        cho2, jung2, jong2 = _parts(out)
        assert jong2 == jong
        assert (cho2 != cho) != (jung2 != jung)


def test_non_hangul_character_cannot_be_substituted():
    assert substitute_jamo("A", random.Random(0)) is None


def test_typos_hit_one_in_five_words_and_leave_everything_else_alone():
    text = "다음 중 세포막 에 대한 설명 으로 옳은 것 은 (가) DNA"
    out = add_typos(text, rate=0.2, rng=random.Random(1))
    words, out_words = text.split(" "), out.split(" ")
    assert len(out_words) == len(words)
    changed = [i for i, (a, b) in enumerate(zip(words, out_words, strict=True)) if a != b]
    assert len(changed) == 2  # 12 words * 0.2, rounded
    for i in changed:
        assert len(out_words[i]) == len(words[i])
        assert sum(a != b for a, b in zip(words[i], out_words[i], strict=True)) == 1
    assert out_words[-1] == "DNA"


def test_typos_are_reproducible_for_the_same_seed():
    text = "다음 중 옳은 것은 무엇인가"
    assert add_typos(text, 0.2, random.Random(7)) == add_typos(text, 0.2, random.Random(7))
