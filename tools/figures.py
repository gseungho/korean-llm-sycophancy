"""Figures for the write-up, drawn from the saved answers.

    python tools/figures.py

Writes PNG files to docs/figures/ and prints the numbers behind each one. Reads shared/trials.csv;
nothing here calls the API. The intervals of the pre-registered comparisons are computed exactly
as in the main report.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ksyco import trial_table, wordings  # noqa: E402
from ksyco.main_report import PRIMARY  # noqa: E402
from ksyco.metrics import contrast, flip_rate  # noqa: E402

TRIALS = "shared/trials.csv"
OUT_DIR = Path("docs/figures")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

ARMS = {
    ("haiku", "on"): ("Haiku 5.5, 생각 켬", BLUE, "o"),
    ("haiku", "off"): ("Haiku 5.5, 생각 끔", ORANGE, "s"),
    ("sonnet", "off"): ("Sonnet 5.5, 생각 끔", AQUA, "D"),
}
WORDINGS = (1, 2, 3)
# Added phrase -> (row label, tick label). The order groups phrases without and with an identity.
PHRASES = {
    "rebut": ("붙인 말 없음", "없음"),
    "filler": ('"저기요, …"', "저기요"),
    "sure": ('"확실한데요, …"', "확실\n한데요"),
    "opinion": ('"제 생각인데요, …"', "제 생각\n인데요"),
    "student": ('"저 고등학생인데요, …"', "고등학생\n인데요"),
    "professor": ('"저 교수인데요, …"', "교수\n인데요"),
    "passerby": ('"저 지나가던 사람인데요, …"', "지나가던\n사람인데요"),
    "titleafter": ('"… 저 교수예요." (뒤에)', "교수예요\n(뒤에)"),
}

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "lines.markeredgecolor": SURFACE,
        "lines.markeredgewidth": 1.5,
        "legend.frameon": False,
    }
)


def conds(family, levels):
    return [f"{family}_{level}_p{p}" for level in levels for p in WORDINGS]


def rate(trials, names):
    return 100 * sum(flip_rate(trials, cond) for cond in names) / len(names)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {OUT_DIR / name}")


def preregistered(groups, cfg):
    """The six pre-registered differences with 99% intervals against the +-5%p band."""
    titles = {
        "speech_level": "하십시오체 반박에서 해체 반박을 뺀 차이",
        "professor": '"교수인데"를 붙인 반박에서 붙이지 않은 반박을 뺀 차이',
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.9), sharex=True)
    for ax, (name, (conds_a, conds_b, _)) in zip(axes, PRIMARY.items(), strict=True):
        ax.axvspan(-5, 5, color=BAND, lw=0)
        ax.axvline(0, color=AXIS, lw=1)
        for y, (arm, (label, _, _)) in enumerate(ARMS.items()):
            point, low, high = (
                100 * v
                for v in contrast(
                    groups[arm],
                    conds_a,
                    conds_b,
                    level=0.99,
                    n_boot=cfg["n_boot"],
                    seed=cfg["seed"],
                )
            )
            print(f"  {name:<13}{label:<22}{point:+6.1f} [{low:+6.1f}, {high:+6.1f}]")
            ax.plot([low, high], [y, y], color=BLUE, solid_capstyle="round")
            ax.plot(point, y, "o", color=BLUE)
            ax.annotate(
                f"{point:+.1f}%p",
                (point, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_yticks(range(len(ARMS)), [label for label, _, _ in ARMS.values()])
        ax.set_ylim(len(ARMS) - 0.45, -0.75)
        ax.set_title(titles[name])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
    axes[1].text(0, 1, "최소 차이\n±5%p 안", ha="center", va="center", fontsize=8.5, color=INK2)
    axes[1].set_xlabel("번복률의 차이 (%p), 선은 99% 구간")
    axes[1].set_xlim(-30, 10)
    fig.tight_layout(h_pad=1.4)
    save(fig, "preregistered.png")


def speech_two_by_two(groups):
    """Formality by honorific: one panel per model setting, the same 12%p span in each."""
    lines = {"격식체 (하십시오체, 해라체)": (("haera", "hasipsio"), BLUE, "o")}
    lines["비격식체 (해요체, 해체)"] = (("hae", "haeyo"), ORANGE, "s")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.3))
    for ax, (arm, (label, _, _)) in zip(axes, ARMS.items(), strict=True):
        values = {}
        for name, (levels, color, marker) in lines.items():
            values[name] = [rate(groups[arm], conds("rebut", [level])) for level in levels]
            ax.plot([0, 1], values[name], marker=marker, color=color, label=name)
            print(f"  {label:<22}{name:<26}{values[name][0]:5.1f} {values[name][1]:5.1f}")
        middle = round(sum(sum(v) for v in values.values()) / 4)
        ax.set_ylim(middle - 6, middle + 6)
        ax.set_yticks(range(middle - 6, middle + 7, 2))
        ax.yaxis.set_major_formatter("{x:.0f}%")
        ax.set_xlim(-0.35, 1.45)
        ax.set_xticks([0, 1], ["낮춤\n해라체\n해체", "높임\n하십시오체\n해요체"], fontsize=8.5)
        ax.set_title(label, fontsize=9.5)
        ax.grid(axis="y")
        ax.tick_params(length=0)
        # A ruler for the smallest difference that was agreed to matter.
        ax.plot([1.3, 1.3], [middle - 5.5, middle - 0.5], color=MUTED, lw=1.2)
        ax.text(1.36, middle - 3, "5%p", fontsize=8, color=INK2, va="center", rotation=90)
    axes[0].set_ylabel("번복률")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.6)
    save(fig, "speech_two_by_two.png")


def phrases(groups):
    """Flip rate by added phrase (polite style, which every model setting has)."""
    rows = [0, 1.4, 2.4, 3.4, 4.8, 5.8, 6.8, 7.8]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for arm, (label, color, marker) in ARMS.items():
        values = [rate(groups[arm], conds(phrase, ["haeyo"])) for phrase in PHRASES]
        print(f"  {label:<22}" + " ".join(f"{v:5.1f}" for v in values))
        ax.plot([values[0], values[0]], [rows[0], rows[-1] + 0.4], color=color, lw=1, alpha=0.35)
        ax.plot(values, rows, marker, color=color, label=label)
        ax.annotate(
            f"{values[0]:.0f}%",
            (values[0], rows[0]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_yticks(rows, [label for label, _ in PHRASES.values()])
    ax.set_ylim(rows[-1] + 0.6, -1.0)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter("{x:.0f}%")
    ax.set_xlabel("번복률 (맞는 첫 답을 반박 뒤에 바꾼 비율), 해요체 반박")
    ax.grid(axis="y")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.text(101, 2.4, "신분이\n없는 말", fontsize=8.5, color=INK2, va="center")
    ax.text(101, 6.3, "신분이\n든 말", fontsize=8.5, color=INK2, va="center")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.45, 1.13), handletextpad=0.3)
    fig.tight_layout()
    save(fig, "phrases.png")


def phrase_by_level(groups):
    """Interaction plot of added phrase and speech level for the two Haiku settings."""
    lines = {"해요체": ("haeyo", BLUE, "o"), "해체": ("hae", ORANGE, "s")}
    arms = [arm for arm in ARMS if arm[0] == "haiku"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    for ax, arm in zip(axes, arms, strict=True):
        for name, (level, color, marker) in lines.items():
            values = [rate(groups[arm], conds(phrase, [level])) for phrase in PHRASES]
            print(f"  {ARMS[arm][0]:<22}{name:<6}" + " ".join(f"{v:5.1f}" for v in values))
            ax.plot(range(len(PHRASES)), values, marker=marker, color=color, label=name)
        ax.set_title(ARMS[arm][0])
        ax.yaxis.set_major_formatter("{x:.0f}%")
        ax.set_ylabel("번복률")
        ax.grid(axis="y")
        ax.tick_params(length=0)
    axes[0].set_ylim(30, 80)
    axes[1].set_ylim(55, 95)
    axes[1].set_xticks(range(len(PHRASES)), [tick for _, tick in PHRASES.values()], fontsize=9)
    axes[0].legend(loc="upper right", ncol=2)
    fig.tight_layout(h_pad=1.2)
    save(fig, "phrase_by_level.png")


def wording_frames(groups):
    """Formal minus plain flip rate of every new wording frame, grouped by predicate."""
    cfg = json.loads(Path("configs/wordings1_v1.json").read_text(encoding="utf-8"))
    subjects = {t["item"]: t["uid"].split(":")[0] for trials in groups.values() for t in trials}
    items = [{"subject": subjects[i]} for i in range(len(subjects))]
    labels = [f"{p['hasipsio'][:-1]} / {p['hae'][:-1]}" for p in wordings.PREDICATES]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.7), sharey=True)
    for ax, (arm, (label, color, marker)) in zip(axes, ARMS.items(), strict=True):
        chosen = wordings.arm_items(items, arm[0], cfg)
        trials = [t for t in groups[arm] if t["rep"] == cfg["rep"] and t["item"] in chosen]
        effects = wordings.analyze(trials, wordings.new_frames(), n_boot=1)["effects"]
        ax.axvspan(-5, 5, color=BAND, lw=0)
        ax.axvline(0, color=AXIS, lw=1)
        for index in range(len(labels)):
            values = sorted(100 * d for frame, d in effects.items() if frame[2] == index)
            offsets = np.linspace(-0.28, 0.28, len(values))
            ax.plot(values, index + offsets, marker, color=color, alpha=0.6, markersize=6)
            ax.plot(np.mean(values), index, "D", color=INK, markersize=7)
            print(f"  {label:<22}{labels[index]:<24}{np.mean(values):+6.1f}")
        ax.set_title(label, fontsize=9.5)
        ax.set_xlim(-20, 20)
        ax.grid(axis="x")
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    axes[1].set_xlabel(
        "하십시오체 번복률 빼기 해체 번복률 (%p). 색 점은 문구 틀 하나, 검은 마름모는 평균"
    )
    fig.tight_layout(w_pad=1.2)
    save(fig, "wording_frames.png")


def main():
    cfg = json.loads(Path("configs/main_v1.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for trial in trial_table.read(TRIALS):
        groups[trial["model"], trial["setting"]].append(trial)
    preregistered(groups, cfg)
    speech_two_by_two(groups)
    phrases(groups)
    phrase_by_level(groups)
    wording_frames(groups)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
