"""The trial-level results as a table that can be shared: item ids and answer letters only.

The saved replies cannot be published because they are built from KMMLU items. This table
keeps what the analysis needs and leaves the item text out. One row is one first answer to the
baseline question, with the second answer under every condition that was run after it. `item`
is the position in our sample (it fixes the order in which items are resampled) and `uid` is
the KMMLU subject and row of that item.
"""

import csv

FIXED = ("model", "setting", "item", "uid", "rep", "first", "answer")


def write(trials, items, path):
    """Save assembled trials as CSV. An empty cell means no usable answer or a condition not run."""
    conds = sorted({cond for t in trials for cond in t["seconds"]})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([*FIXED, *conds])
        for t in sorted(trials, key=lambda t: (t["model"], t["setting"], t["item"], t["rep"])):
            uid = items[t["item"]]["uid"]
            row = [
                t["model"],
                t["setting"],
                t["item"],
                uid,
                t["rep"],
                t["first"] or "",
                t["answer"],
            ]
            writer.writerow(row + [t["seconds"].get(cond) or "" for cond in conds])


def read(path):
    """The table as trials for the analysis functions; `seconds` holds only usable answers."""
    with open(path, encoding="utf-8", newline="") as f:
        trials = []
        for row in csv.DictReader(f):
            trial = {key: row.pop(key) for key in FIXED}
            trial |= {"item": int(trial["item"]), "rep": int(trial["rep"])}
            trial["first"] = trial["first"] or None
            trial["seconds"] = {cond: letter for cond, letter in row.items() if letter}
            trials.append(trial)
        return trials
