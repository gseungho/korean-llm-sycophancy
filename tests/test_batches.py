import pytest

from ksyco import batches


def test_api_uses_the_project_key_even_when_a_general_key_is_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "general-key")
    monkeypatch.setenv("KSYCO_ANTHROPIC_API_KEY", "project-key")
    assert batches.BatchApi().client.api_key == "project-key"


def test_api_refuses_to_start_without_the_project_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "general-key")
    monkeypatch.delenv("KSYCO_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="KSYCO_ANTHROPIC_API_KEY"):
        batches.BatchApi()
