"""Turn collected replies into first answers, trials and check answers.

A reply is the API message as a dict, keyed by the custom_id of its request.
"""

import json

from ksyco.build import parse_id
from ksyco.conditions import check_sentences
from ksyco.extract import extract_letter


def answer_letter(message):
    """The chosen letter, or None when the reply was cut off, refused or has no single letter."""
    if message["stop_reason"] != "end_turn":
        return None
    return extract_letter("".join(b["text"] for b in message["content"] if b["type"] == "text"))


def firsts(first_replies):
    """What the second turn needs from each first reply: its blocks, untouched, and its answer."""
    return {
        custom_id: {"content": message["content"], "letter": answer_letter(message)}
        for custom_id, message in first_replies.items()
    }


def assemble_trials(first_replies, second_replies, items):
    """One trial per baseline first answer, with the second answer under each condition."""
    trials = {}
    for custom_id, message in first_replies.items():
        key = parse_id(custom_id)
        if key["qcond"] != "base":
            continue
        trials[key["model"], key["setting"], key["item"], key["rep"]] = {
            "model": key["model"],
            "setting": key["setting"],
            "item": key["item"],
            "rep": key["rep"],
            "first": answer_letter(message),
            "answer": items[key["item"]]["answer"],
            "seconds": {},
        }
    for custom_id, message in second_replies.items():
        key = parse_id(custom_id)
        if key["stage"] != "s":
            continue
        trial = trials[key["model"], key["setting"], key["item"], key["rep"]]
        trial["seconds"][key["cond"]] = answer_letter(message)
    return list(trials.values())


def assemble_overlay_trials(first_replies, second_replies, items):
    """Trials of the overlay block: a changed question, its first answer, the second answers."""
    trials = {}
    for custom_id, message in second_replies.items():
        key = parse_id(custom_id)
        if key["stage"] != "o":
            continue
        model, setting, qcond, i, rep = (
            key[k] for k in ("model", "setting", "qcond", "item", "rep")
        )
        if (model, setting, qcond, i, rep) not in trials:
            first = first_replies[f"f-{model}-{setting}-{qcond}-{i:03d}-{rep}"]
            trials[model, setting, qcond, i, rep] = {
                "model": model,
                "setting": setting,
                "qcond": qcond,
                "item": i,
                "rep": rep,
                "first": answer_letter(first),
                "answer": items[i]["answer"],
                "seconds": {},
            }
        trials[model, setting, qcond, i, rep]["seconds"][key["cond"]] = answer_letter(message)
    return list(trials.values())


def check_records(check_replies):
    """Each check answer with the speech level and kind of the sentence it was about."""
    sentences = check_sentences()
    rows = []
    for custom_id, message in check_replies.items():
        key = parse_id(custom_id)
        sentence = sentences[key["sentence"]]
        text = next(b["text"] for b in message["content"] if b["type"] == "text")
        rows.append(
            {
                "model": key["model"],
                "setting": key["setting"],
                "level": sentence["level"],
                "kind": sentence["kind"],
                "sentence": key["sentence"],
            }
            | json.loads(text)
        )
    return rows
