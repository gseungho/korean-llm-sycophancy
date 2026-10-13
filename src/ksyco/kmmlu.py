"""Load KMMLU test items (CC BY-ND 4.0: keep the files local, never redistribute edits)."""

import csv
import random
import urllib.request

URL = "https://huggingface.co/datasets/HAERAE-HUB/KMMLU/resolve/main/data/{subject}-test.csv"
_TEXT_COLUMNS = ("question", "A", "B", "C", "D")


def download(subject, data_dir):
    """Fetch one subject's test file into `data_dir` unless it is already there."""
    path = data_dir / f"{subject}-test.csv"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(subject=subject), path)
    return path


def to_item(subject, index, row):
    """One CSV row as an item; `index` is the row position in the file and names the item."""
    return {
        "uid": f"{subject}:{index}",
        "subject": subject,
        "question": row["question"].strip(),
        "choices": {k: row[k].strip() for k in "ABCD"},
        "answer": "ABCD"[int(row["answer"]) - 1],
    }


def load_subject(subject, path):
    """Items of one subject file, without rows that lack a question or a choice."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        to_item(subject, i, row)
        for i, row in enumerate(rows)
        if all(row[k].strip() for k in _TEXT_COLUMNS)
    ]


def without(items_by_subject, uids):
    """The same pool minus the items whose uid is in `uids` (items already used elsewhere)."""
    return {
        subject: [item for item in items if item["uid"] not in uids]
        for subject, items in items_by_subject.items()
    }


def sample_items(items_by_subject, n, seed):
    """`n` random items per subject, subject by subject in the order given."""
    sample = []
    for subject, items in items_by_subject.items():
        sample += random.Random(f"{seed}:{subject}").sample(items, n)
    return sample
