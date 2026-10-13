"""Numbers for the pilot report: the quantities of spec section 4 that section 8 decides on."""

from collections import defaultdict
from statistics import mean

from ksyco.build import parse_id
from ksyco.conditions import CONTROLS, DOUBTS, HONORIFIC, REBUTTALS
from ksyco.metrics import (
    agreement,
    flip_rate,
    level_rate,
    required_items,
    resample_baseline,
    wording_range,
)
from ksyco.records import answer_letter

# Dollars per million tokens (input, output). The Batch API halves both.
PRICES = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
BATCH_DISCOUNT = 0.5


def _ratio(numerator, denominator):
    return None if numerator is None or not denominator else numerator / denominator


def summarize(trials):
    """Section 8 quantities for the trials of one model and thinking setting."""
    conds = {c for t in trials for c in t["seconds"]}
    levels = [level for level in REBUTTALS if f"rebut_{level}_p1" in conds]
    answers = [t["first"] for t in trials] + [a for t in trials for a in t["seconds"].values()]
    firsts_by_item = defaultdict(list)
    for t in trials:
        firsts_by_item[t["item"]].append(t["first"])
    baselines = [resample_baseline(firsts) for firsts in firsts_by_item.values()]
    baselines = [b for b in baselines if b is not None]
    level_rates = {level: level_rate(trials, level) for level in levels}
    control_rates = {
        level: flip_rate(trials, f"control_{level}")
        for level in CONTROLS
        if f"control_{level}" in conds
    }
    return {
        "trials": len(trials),
        "first_accuracy": mean(t["first"] == t["answer"] for t in trials),
        "agreement": mean(agreement(firsts) for firsts in firsts_by_item.values()),
        "resample_baseline": mean(baselines) if baselines else None,
        "extraction_failure": answers.count(None) / len(answers),
        "level_rate": level_rates,
        "wording_range": {level: wording_range(trials, level) for level in levels},
        "control_rate": control_rates,
        "control_ratio": {
            level: _ratio(rate, level_rates.get(level)) for level, rate in control_rates.items()
        },
        "doubt_rate": {
            level: flip_rate(trials, f"doubt_{level}")
            for level in DOUBTS
            if f"doubt_{level}" in conds
        },
        "other_rate": {
            cond: flip_rate(trials, cond)
            for cond in sorted(conds)
            if not cond.startswith(("rebut_", "control_", "doubt_"))
        },
    }


def check_summary(rows):
    """Per model: how each speech level was read, and how often the professor was recognised."""
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summary = {}
    for model, model_rows in by_model.items():
        levels = {}
        for level in REBUTTALS:
            group = [r for r in model_rows if r["level"] == level and r["kind"] != "professor"]
            if not group:
                continue
            # English has no speech level, so there is no reference answer to match.
            expected = {True: "높임", False: "높이지 않음", None: None}[HONORIFIC.get(level)]
            levels[level] = {
                "answers": len(group),
                "honorific_match": expected and mean(r["honorific"] == expected for r in group),
                "speaker_superior": mean(r["superior"] == "말하는 사람" for r in group),
                "common": mean(r["common"] == "예" for r in group),
                "authority": mean(r["authority"] for r in group),
            }
        professor = [r for r in model_rows if r["kind"] == "professor"]
        summary[model] = {
            "levels": levels,
            "professor_rate": mean("교수" in r["occupation"] for r in professor)
            if professor
            else None,
        }
    return summary


def mean_usage(replies):
    """Average input and output tokens per call, by (model, setting, stage)."""
    groups = defaultdict(list)
    for custom_id, message in replies.items():
        key = parse_id(custom_id)
        groups[key["model"], key["setting"], key["stage"]].append(message["usage"])
    return {
        group: {
            "calls": len(usages),
            "input": mean(u["input_tokens"] for u in usages),
            "output": mean(u["output_tokens"] for u in usages),
        }
        for group, usages in groups.items()
    }


def batch_cost(replies):
    """Dollars spent per model for these replies at Batch API prices."""
    tokens = defaultdict(lambda: [0, 0])
    for custom_id, message in replies.items():
        model = parse_id(custom_id)["model"]
        tokens[model][0] += message["usage"]["input_tokens"]
        tokens[model][1] += message["usage"]["output_tokens"]
    return {
        model: round(
            (used[0] * PRICES[model][0] + used[1] * PRICES[model][1]) / 1e6 * BATCH_DISCOUNT, 4
        )
        for model, used in tokens.items()
    }


def question_axis(first_replies, items):
    """First-answer accuracy and agreement for each (model, setting, question condition)."""
    letters = defaultdict(lambda: defaultdict(list))
    for custom_id, message in first_replies.items():
        key = parse_id(custom_id)
        group = key["model"], key["setting"], key["qcond"]
        letters[group][key["item"]].append(answer_letter(message))
    axis = {}
    for group, by_item in letters.items():
        answers = [(item, letter) for item, found in by_item.items() for letter in found]
        axis[group] = {
            "answers": len(answers),
            "accuracy": mean(letter == items[item]["answer"] for item, letter in answers),
            "agreement": mean(agreement(found) for found in by_item.values()),
        }
    return axis


LEVEL_NAMES = {
    "hasipsio": "하십시오체",
    "haeyo": "해요체",
    "hae": "해체",
    "haera": "해라체",
    "hao": "하오체",
    "hage": "하게체",
    "english": "영어",
}
SETTING_NAMES = {"on": "생각 켬", "off": "생각 끔"}
STAGE_NAMES = {"c": "화계 확인", "f": "첫 답", "s": "둘째 답"}


def _pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def render(check, summaries, diffs, axis, usage, cost):
    """The pilot report as markdown. `summaries` and `diffs` are keyed by (model, setting)."""
    out = [
        "# 파일럿 결과 요약",
        "",
        "탐색 결과다. 가설이 맞았다는 근거로 쓰지 않는다. 판단 기준은 설계서 8절에 있다.",
        "",
        "## 1. 화계 확인",
        "",
    ]
    for model, summary in check.items():
        rows = [
            [
                LEVEL_NAMES[level],
                s["answers"],
                _pct(s["honorific_match"]),
                _pct(s["speaker_superior"]),
                _pct(s["common"]),
                f"{s['authority']:.1f}",
            ]
            for level, s in summary["levels"].items()
        ]
        header = [
            "화계",
            "답 수",
            "문법 표와 일치",
            "말하는 사람이 윗사람",
            "흔한 말투",
            "권위 (1~5)",
        ]
        out += [f"### {model}", ""] + _table(header, rows)
        out += [
            f"교수 호칭 문장에서 직업을 교수로 답한 비율: {_pct(summary['professor_rate'])}",
            "",
        ]

    out += ["## 2. 번복률", ""]
    for (model, setting), s in summaries.items():
        out += [
            f"### {model}, {SETTING_NAMES[setting]}",
            "",
            f"- 시행 {s['trials']}개, 첫 답 정확도 {_pct(s['first_accuracy'])}, "
            f"첫 답 일치율 {_pct(s['agreement'])}",
            f"- 반복 샘플링 기준선 {_pct(s['resample_baseline'])}, "
            f"추출 실패율 {_pct(s['extraction_failure'])}",
            "",
        ]
        rows = []
        for level, rate in s["level_rate"].items():
            ratio = s["control_ratio"].get(level)
            rows.append(
                [
                    LEVEL_NAMES[level],
                    _pct(rate),
                    _pct(s["control_rate"].get(level)),
                    "-" if ratio is None else f"{ratio:.2f}",
                    _pct(s["wording_range"][level]),
                    _pct(s["doubt_rate"].get(level)),
                ]
            )
        header = [
            "반박의 화계",
            "번복률",
            "같은 화계 재질문의 번복률",
            "재질문 ÷ 반박",
            "문구 간 차이",
            "되묻기의 번복률",
        ]
        out += _table(header, rows)
        if s["other_rate"]:
            others = [[cond, _pct(rate)] for cond, rate in s["other_rate"].items()]
            out += _table(["그 밖의 조건", "번복률"], others)
        if (model, setting) in diffs:
            point, low, high = diffs[model, setting]
            half = (high - low) / 2
            out += [
                f"하십시오체 빼기 해체: {100 * point:+.1f}%p "
                f"(95% 구간 {100 * low:+.1f}%p ~ {100 * high:+.1f}%p, "
                f"폭의 절반 {100 * half:.1f}%p). "
                f"이 폭이면 본 실험에 필요한 문항 수는 {required_items(half)}개다.",
                "",
            ]

    if axis:
        rows = [
            [
                model,
                SETTING_NAMES[setting],
                qcond,
                a["answers"],
                _pct(a["accuracy"]),
                _pct(a["agreement"]),
            ]
            for (model, setting, qcond), a in sorted(axis.items())
        ]
        header = ["모델", "설정", "질문 조건", "답 수", "정확도", "일치율"]
        out += ["## 3. 질문 축의 첫 답", ""] + _table(header, rows)
    if usage:
        rows = [
            [
                model,
                SETTING_NAMES[setting],
                STAGE_NAMES[stage],
                u["calls"],
                f"{u['input']:.0f}",
                f"{u['output']:.0f}",
            ]
            for (model, setting, stage), u in sorted(usage.items())
        ]
        header = ["모델", "설정", "단계", "호출 수", "호출당 입력 token", "호출당 출력 token"]
        out += ["## 4. token과 비용", ""] + _table(header, rows)
        out += [
            f"- {model}: ${dollars:.2f} (Batch API 가격)" for model, dollars in cost.items()
        ] + [""]
    return "\n".join(out)
