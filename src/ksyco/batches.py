"""Thin wrapper over the Message Batches API. This is the only module that spends money."""

import os
import time

import anthropic

# A key of its own for this project, so that runs are billed to a known account and other tools
# that read ANTHROPIC_API_KEY never pick it up. There is no fallback to ANTHROPIC_API_KEY.
KEY_VARIABLE = "KSYCO_ANTHROPIC_API_KEY"


class BatchApi:
    def __init__(self):
        key = os.environ.get(KEY_VARIABLE)
        if not key:
            raise RuntimeError(
                f"환경 변수 {KEY_VARIABLE}가 없습니다. 이 프로젝트 전용 API 키를 만들어 "
                "이 이름으로 설정한 뒤 터미널을 다시 여세요."
            )
        # The key is never printed or stored.
        self.client = anthropic.Anthropic(api_key=key)

    def submit(self, requests):
        return self.client.messages.batches.create(requests=requests).id

    def wait(self, batch_id, poll_seconds=60):
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            counts = batch.request_counts
            print(f"  {batch_id}: 처리 중 {counts.processing}개, 끝남 {counts.succeeded}개")
            time.sleep(poll_seconds)

    def collect(self, batch_id):
        """(replies, failures): replies of succeeded requests as dicts, and the rest as returned."""
        replies, failures = {}, {}
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                # exclude_none drops unset optional fields only; an empty thinking text is kept,
                # which the API needs when the reply is sent back in the second turn.
                message = result.result.message.model_dump(mode="json", exclude_none=True)
                replies[result.custom_id] = message
            else:
                failures[result.custom_id] = result.result.model_dump(
                    mode="json", exclude_none=True
                )
        return replies, failures
