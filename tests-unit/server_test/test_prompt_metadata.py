"""Tests for the per-prompt metadata registry used to propagate ``workflow_id``
through WebSocket events without coupling ``execution.py`` to workflow-level
concepts.

The registry is a dict on ``PromptServer`` (keyed by ``prompt_id``) registered
at submission time (``post_prompt``), merged onto outbound payloads in
``send_sync`` via ``merge_prompt_metadata``, and dropped in ``main.py`` *after*
the terminal ``executing: {node: None}`` send so the final frame carries the
same ``workflow_id`` as the rest of the prompt.
"""

import threading

import pytest

from comfy_execution.metadata import (
    build_prompt_metadata,
    merge_prompt_metadata,
)


@pytest.fixture
def registry():
    return {}


@pytest.fixture
def lock():
    return threading.Lock()


class TestBuildPromptMetadata:
    def test_returns_workflow_id_when_present(self):
        extra_data = {"extra_pnginfo": {"workflow": {"id": "wf-1"}}}
        assert build_prompt_metadata(extra_data) == {"workflow_id": "wf-1"}

    def test_empty_when_workflow_id_missing(self):
        assert build_prompt_metadata({}) == {}
        assert build_prompt_metadata({"extra_pnginfo": {}}) == {}
        assert build_prompt_metadata({"extra_pnginfo": {"workflow": {}}}) == {}

    def test_empty_when_workflow_id_not_a_non_empty_string(self):
        assert build_prompt_metadata({"extra_pnginfo": {"workflow": {"id": ""}}}) == {}
        assert build_prompt_metadata({"extra_pnginfo": {"workflow": {"id": 42}}}) == {}
        assert build_prompt_metadata({"extra_pnginfo": {"workflow": {"id": None}}}) == {}

    def test_empty_on_non_dict_input(self):
        assert build_prompt_metadata(None) == {}
        assert build_prompt_metadata("not a dict") == {}


class TestMergeMetadata:
    """``merge_prompt_metadata`` is the transparent injection point used by
    ``PromptServer.send_sync``. Event payload fields win on conflict so the
    executor can never be overridden by stale registry data, and binary payloads
    (the preview tuple) pass through untouched."""

    def test_merges_workflow_id_when_prompt_id_known(self, registry, lock):
        registry["p1"] = {"workflow_id": "wf-1"}
        merged = merge_prompt_metadata(registry, lock, {"node": "n1", "prompt_id": "p1"})
        assert merged == {"node": "n1", "prompt_id": "p1", "workflow_id": "wf-1"}

    def test_passthrough_when_prompt_id_unknown(self, registry, lock):
        merged = merge_prompt_metadata(registry, lock, {"node": "n1", "prompt_id": "missing"})
        assert merged == {"node": "n1", "prompt_id": "missing"}

    def test_passthrough_when_no_prompt_id(self, registry, lock):
        registry["p1"] = {"workflow_id": "wf-1"}
        merged = merge_prompt_metadata(registry, lock, {"status": {"queue_remaining": 0}})
        assert merged == {"status": {"queue_remaining": 0}}

    def test_passthrough_for_non_dict_payload(self, registry, lock):
        registry["p1"] = {"workflow_id": "wf-1"}
        binary = (b"image-bytes", {"prompt_id": "p1"})
        assert merge_prompt_metadata(registry, lock, binary) is binary

    def test_event_payload_wins_over_registered_metadata(self, registry, lock):
        registry["p1"] = {"workflow_id": "wf-registered"}
        merged = merge_prompt_metadata(
            registry, lock, {"prompt_id": "p1", "workflow_id": "wf-caller"}
        )
        assert merged["workflow_id"] == "wf-caller"


class TestRaceRegressionForTerminalExecutingFrame:
    """Regression for the PR #13684 finally-clear race.

    In the reverted PR, the executor's ``finally`` cleared ``last_workflow_id``
    before ``main.py`` emitted the post-completion ``executing: {node: None}``
    frame — so that terminal frame shipped ``workflow_id=None``.

    With the registry approach, ``main.py`` unregisters *after* the terminal
    send, so the merge sees the registered metadata. This test simulates that
    ordering to lock in the contract.
    """

    def test_terminal_executing_frame_includes_workflow_id(self, registry, lock):
        registry["p1"] = {"workflow_id": "wf-1"}

        # main.py emits the terminal frame BEFORE unregistering.
        terminal = merge_prompt_metadata(registry, lock, {"node": None, "prompt_id": "p1"})
        registry.pop("p1", None)  # main.py's finally: unregister_prompt_metadata

        assert terminal == {"node": None, "prompt_id": "p1", "workflow_id": "wf-1"}

        # After unregister, any straggler events emitted by extensions after
        # completion are no longer decorated. Verifies the registry is actually
        # released, not just shadowed.
        straggler = merge_prompt_metadata(registry, lock, {"node": None, "prompt_id": "p1"})
        assert "workflow_id" not in straggler
