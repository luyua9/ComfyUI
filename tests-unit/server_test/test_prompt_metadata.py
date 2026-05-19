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
    resolve_progress_text_sid,
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
        registry["p1"] = [{"workflow_id": "wf-1"}]
        merged = merge_prompt_metadata(registry, lock, {"node": "n1", "prompt_id": "p1"})
        assert merged == {"node": "n1", "prompt_id": "p1", "workflow_id": "wf-1"}

    def test_passthrough_when_prompt_id_unknown(self, registry, lock):
        merged = merge_prompt_metadata(registry, lock, {"node": "n1", "prompt_id": "missing"})
        assert merged == {"node": "n1", "prompt_id": "missing"}

    def test_passthrough_when_no_prompt_id(self, registry, lock):
        registry["p1"] = [{"workflow_id": "wf-1"}]
        merged = merge_prompt_metadata(registry, lock, {"status": {"queue_remaining": 0}})
        assert merged == {"status": {"queue_remaining": 0}}

    def test_passthrough_for_non_dict_payload(self, registry, lock):
        registry["p1"] = [{"workflow_id": "wf-1"}]
        binary = (b"image-bytes", {"prompt_id": "p1"})
        assert merge_prompt_metadata(registry, lock, binary) is binary

    def test_event_payload_wins_over_registered_metadata(self, registry, lock):
        registry["p1"] = [{"workflow_id": "wf-registered"}]
        merged = merge_prompt_metadata(
            registry, lock, {"prompt_id": "p1", "workflow_id": "wf-caller"}
        )
        assert merged["workflow_id"] == "wf-caller"


class TestProgressTextSidResolution:
    """``BinaryEventTypes.TEXT`` frames don't yet carry ``prompt_id`` /
    ``workflow_id`` in their wire shape, so cross-client routing has to happen
    at the ``sid`` level instead of via the metadata merge. The default sid
    pins the broadcast to the active prompt's client.
    """

    def test_explicit_sid_passes_through(self):
        assert resolve_progress_text_sid("client-explicit", "client-active") == "client-explicit"

    def test_none_sid_defaults_to_active_client(self):
        assert resolve_progress_text_sid(None, "client-active") == "client-active"

    def test_none_sid_with_no_active_client_stays_none(self):
        # No active prompt means there is no sensible recipient — fall through
        # to broadcast rather than fabricate a target.
        assert resolve_progress_text_sid(None, None) is None


class TestPromptIdCollision:
    """Two prompts may be submitted with the same ``prompt_id`` (client retry,
    forced custom id, partner-integration deduplication, etc.). With a flat
    dict-keyed registry the second registration would clobber the first and a
    single ``unregister`` call would erase metadata still needed by the other
    prompt. The stack-based registry resolves both cases."""

    def test_duplicate_register_does_not_clobber_prior_entry(self, registry, lock):
        # Caller B clobbers A in the merge view (last-wins), but A's metadata
        # is still in the stack and reappears after B unregisters.
        registry.setdefault("p1", []).append({"workflow_id": "wf-A"})
        registry["p1"].append({"workflow_id": "wf-B"})

        merged = merge_prompt_metadata(registry, lock, {"prompt_id": "p1"})
        assert merged["workflow_id"] == "wf-B"

        registry["p1"].pop()

        merged = merge_prompt_metadata(registry, lock, {"prompt_id": "p1"})
        assert merged["workflow_id"] == "wf-A"

    def test_single_unregister_does_not_drop_concurrent_submission(self, registry, lock):
        registry.setdefault("p1", []).append({"workflow_id": "wf-A"})
        registry["p1"].append({"workflow_id": "wf-B"})

        # Only one of the two prompts finished — pop once.
        registry["p1"].pop()

        merged = merge_prompt_metadata(registry, lock, {"prompt_id": "p1"})
        assert "workflow_id" in merged

    def test_full_drain_clears_registry(self, registry, lock):
        registry.setdefault("p1", []).append({"workflow_id": "wf-A"})
        registry["p1"].append({"workflow_id": "wf-B"})
        registry["p1"].pop()
        registry["p1"].pop()

        merged = merge_prompt_metadata(registry, lock, {"prompt_id": "p1"})
        assert "workflow_id" not in merged


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
        registry["p1"] = [{"workflow_id": "wf-1"}]

        # main.py emits the terminal frame BEFORE unregistering.
        terminal = merge_prompt_metadata(registry, lock, {"node": None, "prompt_id": "p1"})
        registry["p1"].pop()
        if not registry["p1"]:
            registry.pop("p1", None)

        assert terminal == {"node": None, "prompt_id": "p1", "workflow_id": "wf-1"}

        # After unregister, any straggler events emitted by extensions after
        # completion are no longer decorated. Verifies the registry is actually
        # released, not just shadowed.
        straggler = merge_prompt_metadata(registry, lock, {"node": None, "prompt_id": "p1"})
        assert "workflow_id" not in straggler
