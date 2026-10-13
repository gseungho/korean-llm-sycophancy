"""Jamo-level typos: swap one jamo for the key next to it on a two-set (dubeolsik) keyboard."""

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
KEY_ROWS = ["ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ", "ㅁㄴㅇㄹㅎㅗㅓㅏㅣ", "ㅋㅌㅊㅍㅠㅜㅡ"]


def neighbors(jamo):
    """Keys directly left or right of `jamo` that are the same kind (consonant or vowel)."""
    same_kind = CHO if jamo in CHO else JUNG
    for row in KEY_ROWS:
        i = row.find(jamo)
        if i >= 0:
            return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row) and row[j] in same_kind]
    return []


def _substitutions(char):
    """Every syllable reachable from `char` by one adjacent-key swap of its initial or vowel."""
    code = ord(char) - 0xAC00
    if not 0 <= code < 11172:
        return []
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    pairs = [(CHO.index(c), jung) for c in neighbors(CHO[cho])]
    pairs += [(cho, JUNG.index(j)) for j in neighbors(JUNG[jung])]
    return [chr(0xAC00 + c * 588 + j * 28 + jong) for c, j in pairs]


def substitute_jamo(char, rng):
    """One random adjacent-key typo of a Hangul syllable, or None if `char` cannot take one."""
    options = _substitutions(char)
    return rng.choice(options) if options else None


def add_typos(text, rate, rng):
    """Put one typo into `rate` of the space-separated words (at least one word)."""
    words = text.split(" ")
    spots = {
        w: [c for c, ch in enumerate(word) if _substitutions(ch)] for w, word in enumerate(words)
    }
    spots = {w: chars for w, chars in spots.items() if chars}
    count = min(len(spots), max(1, round(len(words) * rate)))
    for w in rng.sample(sorted(spots), count):
        c = rng.choice(spots[w])
        words[w] = words[w][:c] + substitute_jamo(words[w][c], rng) + words[w][c + 1 :]
    return " ".join(words)
