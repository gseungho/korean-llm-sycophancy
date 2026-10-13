import json

from ksyco import pilot


class FakeApi:
    """Stands in for the paid API: records submissions and answers every request with "B"."""

    def __init__(self):
        self.submitted = []

    def submit(self, requests):
        self.submitted.append([r["custom_id"] for r in requests])
        return f"batch_{len(self.submitted)}"

    def wait(self, batch_id):
        pass

    def collect(self, batch_id):
        ids = self.submitted[int(batch_id.split("_")[1]) - 1]
        ok = {
            i: {"content": [{"type": "text", "text": "B"}], "stop_reason": "end_turn"} for i in ids
        }
        return ok, {}


REQUESTS = [{"custom_id": f"f-haiku-on-base-{i:03d}-0", "params": {}} for i in range(5)]


def test_stage_submits_in_chunks_and_saves_every_reply(tmp_path):
    api = FakeApi()
    replies = pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=2)
    assert [len(chunk) for chunk in api.submitted] == [2, 2, 1]
    assert set(replies) == {r["custom_id"] for r in REQUESTS}
    assert pilot.load_replies(tmp_path / "first.jsonl") == replies


def test_finished_stage_is_not_sent_again(tmp_path):
    pilot.run_stage("first", REQUESTS, tmp_path, FakeApi(), chunk_size=2)
    api = FakeApi()
    pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=2)
    assert api.submitted == []


def test_interrupted_stage_collects_its_saved_batches_without_paying_twice(tmp_path):
    api = FakeApi()
    api.submit(REQUESTS)  # the batch was sent, then the run died before collecting
    (tmp_path / "state.json").write_text(json.dumps({"first": ["batch_1"]}), encoding="utf-8")
    replies = pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=10)
    assert len(api.submitted) == 1
    assert len(replies) == 5


def test_stage_interrupted_between_chunks_sends_only_the_missing_chunks(tmp_path):
    api = FakeApi()
    api.submit(REQUESTS[:2])  # the first of three chunks went out before the run died
    (tmp_path / "state.json").write_text(json.dumps({"first": ["batch_1"]}), encoding="utf-8")
    replies = pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=2)
    assert [len(chunk) for chunk in api.submitted] == [2, 2, 1]
    assert len(replies) == 5


def test_plan_counts_match_the_spec():
    cfg = json.loads(open("configs/pilot_v1.json", encoding="utf-8").read())
    assert pilot.plan(cfg) == {
        "check": 1160,
        "first": 2400 + 390,
        "second_min": 13800 + 6900 + 2070,
        "second_max": 15600 + 7800 + 2340,
    }


def test_report_stage_writes_a_report_from_the_saved_replies(tmp_path):
    def reply(text):
        usage = {"input_tokens": 100, "output_tokens": 10}
        content = [{"type": "text", "text": text}]
        return {"content": content, "stop_reason": "end_turn", "usage": usage}

    def save(name, replies):
        lines = [json.dumps({"custom_id": k, "message": v}) for k, v in replies.items()]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    items = [{"uid": "Law:0", "answer": "B"}]
    (tmp_path / "items.json").write_text(json.dumps(items), encoding="utf-8")
    answer = {"honorific": "높임", "superior": "비슷함", "common": "예", "authority": 3}
    save("check", {"c-haiku-on-00-0": reply(json.dumps(answer | {"occupation": "밝히지 않음"}))})
    save("first", {"f-haiku-on-base-000-0": reply("B")})
    second = {"s-haiku-on-000-0-rebut_hasipsio_p1": "A", "s-haiku-on-000-0-rebut_hae_p1": "B"}
    save("second", {k: reply(v) for k, v in second.items()})

    text = pilot.write_report({"seed": 1}, tmp_path)

    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text
    assert "+100.0%p" in text  # the formal rebuttal flipped the answer, the plain one did not


def test_stage_can_be_sent_without_waiting_and_collected_in_a_later_run(tmp_path):
    api = FakeApi()
    assert pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=10, wait=False) is None
    assert not (tmp_path / "first.jsonl").exists()

    replies = pilot.run_stage("first", REQUESTS, tmp_path, api, chunk_size=10)

    assert len(api.submitted) == 1  # the later run only collected
    assert len(replies) == 5
