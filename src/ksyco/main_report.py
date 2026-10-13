"""Analysis of the main experiment (docs/specs/2026-10-13-main.md, section 5)."""

from collections import defaultdict
from statistics import mean

from ksyco.main_design import JOBS, OVERLAY_CONDITIONS, rebuttal_conditions
from ksyco.metrics import agreement, contrast, flip_rate, resample_baseline, verdict


def _wordings(prefix):
    return [f"{prefix}_p{p}" for p in (1, 2, 3)]


UNTITLED = _wordings("rebut_haeyo") + _wordings("rebut_hae")
PROFESSOR = _wordings("professor_haeyo") + _wordings("professor_hae")
STUDENT = _wordings("student_haeyo") + _wordings("student_hae")
KOREAN = _wordings("rebut_hasipsio") + UNTITLED

# Pre-registered comparisons: (conditions, compared with, predicted direction). Judged with a
# 99% interval because there are six of them (two comparisons in three model settings).
PRIMARY = {
    "speech_level": (_wordings("rebut_hasipsio"), _wordings("rebut_hae"), +1),
    "professor": (PROFESSOR, UNTITLED, -1),
}
# Exploratory comparisons: (conditions, compared with). 95% interval, no verdict.
EXPLORATORY = {
    "formality": (_wordings("rebut_hasipsio"), _wordings("rebut_haeyo")),
    "honorific": (_wordings("rebut_haeyo"), _wordings("rebut_hae")),
    "professor_minus_student": (PROFESSOR, STUDENT),
    "student": (STUDENT, UNTITLED),
    **{f"job_{job}": ([f"job_{job}"], ["rebut_haeyo_p1"]) for job in JOBS},
    "quote_professor": (["quote_professor"], ["rebut_haeyo_p1"]),
    "haera": (_wordings("rebut_haera"), _wordings("rebut_hae")),
    "hage": (_wordings("rebut_hage"), _wordings("rebut_hae")),
    "english": (_wordings("rebut_english"), KOREAN),
    "choseong": (["choseong"], ["rebut_hae_p1"]),
    "typo_hae": (["typo_hae"], ["rebut_hae_p1"]),
}


def analyze(trials, n_boot=10000, seed=0):
    """Everything reported for the baseline-question trials of one model and thinking setting."""
    primary = {}
    for name, (conds_a, conds_b, direction) in PRIMARY.items():
        point, low, high = contrast(trials, conds_a, conds_b, level=0.99, n_boot=n_boot, seed=seed)
        outcome = verdict(point, low, high, direction)
        primary[name] = {"point": point, "low": low, "high": high, "verdict": outcome}
    exploratory = {}
    for name, (conds_a, conds_b) in EXPLORATORY.items():
        point, low, high = contrast(trials, conds_a, conds_b, n_boot=min(n_boot, 2000), seed=seed)
        exploratory[name] = {"point": point, "low": low, "high": high}

    firsts_by_item = defaultdict(list)
    for t in trials:
        firsts_by_item[t["item"]].append(t["first"])
    baselines = [resample_baseline(firsts) for firsts in firsts_by_item.values()]
    baselines = [b for b in baselines if b is not None]
    answers = [t["first"] for t in trials] + [a for t in trials for a in t["seconds"].values()]
    return {
        "trials": len(trials),
        "first_accuracy": mean(t["first"] == t["answer"] for t in trials),
        "agreement": mean(agreement(firsts) for firsts in firsts_by_item.values()),
        "resample_baseline": mean(baselines) if baselines else None,
        "unusable": answers.count(None) / len(answers),
        "rates": {cond: flip_rate(trials, cond) for cond in rebuttal_conditions()},
        "primary": primary,
        "exploratory": exploratory,
    }


def overlay_summary(overlay_trials):
    """Flip rate of each overlay condition by (model, setting, question form)."""
    groups = defaultdict(list)
    for t in overlay_trials:
        groups[t["model"], t["setting"], t["qcond"]].append(t)
    return {
        group: {cond: flip_rate(trials, cond) for cond in OVERLAY_CONDITIONS}
        for group, trials in sorted(groups.items())
    }


SETTING_NAMES = {"on": "생각 켬", "off": "생각 끔"}
VERDICT_NAMES = {
    "supported": "예측이 맞음",
    "below_min": "차이가 최소 차이보다 작음",
    "reversed": "반대 방향",
    "inconclusive": "판정 보류",
}
PRIMARY_NAMES = {
    "speech_level": "하십시오체 빼기 해체",
    "professor": "교수 호칭 있음 빼기 없음",
}
EXPLORATORY_NAMES = {
    "formality": "하십시오체 빼기 해요체 (격식)",
    "honorific": "해요체 빼기 해체 (높임)",
    "professor_minus_student": "교수 빼기 고등학생",
    "student": "고등학생 호칭 있음 빼기 없음",
    "job_doctor": "의사 빼기 호칭 없음 (문구 1)",
    "job_judge": "판사 빼기 호칭 없음 (문구 1)",
    "job_pharmacist": "약사 빼기 호칭 없음 (문구 1)",
    "job_lawmaker": "국회의원 빼기 호칭 없음 (문구 1)",
    "job_teacher": "교사 빼기 호칭 없음 (문구 1)",
    "quote_professor": "교수님 인용 빼기 호칭 없음 (문구 1)",
    "haera": "해라체 빼기 해체",
    "hage": "하게체 빼기 해체",
    "english": "영어 빼기 한국어 세 화계",
    "choseong": "초성체 빼기 해체 (문구 1)",
    "typo_hae": "오타가 든 해체 빼기 해체 (문구 1)",
}


def _pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _diff(result):
    if result["point"] is None:
        return "-", "-"
    interval = f"{100 * result['low']:+.1f}%p ~ {100 * result['high']:+.1f}%p"
    return f"{100 * result['point']:+.1f}%p", interval


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def render(results, axis, overlay, usage, cost):
    """The main-experiment report as markdown. `results` is keyed by (model, setting)."""
    arms = [f"{model}, {SETTING_NAMES[setting]}" for model, setting in results]
    out = [
        "# 본 실험 결과",
        "",
        "판정 규칙은 설계서 5절에 있다.",
        "",
        "## 1. 미리 정한 비교 (99% 구간)",
        "",
    ]
    rows = []
    for arm, result in zip(arms, results.values(), strict=True):
        for name, comparison in result["primary"].items():
            point, interval = _diff(comparison)
            rows.append(
                [arm, PRIMARY_NAMES[name], point, interval, VERDICT_NAMES[comparison["verdict"]]]
            )
    out += _table(["모델과 설정", "비교", "차이", "99% 구간", "판정"], rows)

    out += ["## 2. 설정별 요약", ""]
    rows = [
        [
            arm,
            r["trials"],
            _pct(r["first_accuracy"]),
            _pct(r["agreement"]),
            _pct(r["resample_baseline"]),
            _pct(r["unusable"]),
        ]
        for arm, r in zip(arms, results.values(), strict=True)
    ]
    header = [
        "모델과 설정",
        "시행",
        "첫 답 정확도",
        "첫 답 일치율",
        "반복 샘플링 기준선",
        "못 쓰는 답",
    ]
    out += _table(header, rows)

    out += ["## 3. 탐색 비교 (95% 구간, 판정 없음)", ""]
    for arm, result in zip(arms, results.values(), strict=True):
        rows = [[EXPLORATORY_NAMES[name], *_diff(c)] for name, c in result["exploratory"].items()]
        out += [f"### {arm}", ""] + _table(["비교", "차이", "95% 구간"], rows)

    out += ["## 4. 조건별 번복률", ""]
    conds = list(rebuttal_conditions())
    rows = [[cond, *(_pct(r["rates"][cond]) for r in results.values())] for cond in conds]
    out += _table(["조건", *arms], rows)

    if axis:
        rows = [
            [
                model,
                SETTING_NAMES[setting],
                form,
                a["answers"],
                _pct(a["accuracy"]),
                _pct(a["agreement"]),
            ]
            for (model, setting, form), a in sorted(axis.items())
        ]
        header = ["모델", "설정", "질문 형태", "답 수", "정확도", "일치율"]
        out += ["## 5. 질문 축의 첫 답", ""] + _table(header, rows)
    if overlay:
        rows = [
            [
                model,
                SETTING_NAMES[setting],
                form,
                *(_pct(rates[cond]) for cond in OVERLAY_CONDITIONS),
            ]
            for (model, setting, form), rates in overlay.items()
        ]
        header = ["모델", "설정", "질문 형태", *OVERLAY_CONDITIONS]
        out += ["## 6. 질문을 바꾼 뒤의 번복률", ""] + _table(header, rows)
    if usage:
        rows = [
            [
                model,
                SETTING_NAMES[setting],
                stage,
                u["calls"],
                f"{u['input']:.0f}",
                f"{u['output']:.0f}",
            ]
            for (model, setting, stage), u in sorted(usage.items())
        ]
        header = ["모델", "설정", "단계", "호출 수", "호출당 입력 token", "호출당 출력 token"]
        out += ["## 7. token과 비용", ""] + _table(header, rows)
        out += [
            f"- {model}: ${dollars:.2f} (Batch API 가격)" for model, dollars in cost.items()
        ] + [""]
    return "\n".join(out)
