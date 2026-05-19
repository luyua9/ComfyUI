"""Tests for the per-prompt metadata registry used to propagate ``workflow_id``
through WebSocket events without coupling ``execution.py`` to workflow-level
concepts.

The registry is keyed by an internal monotonic token (NOT by ``prompt_id``)
because ``post_prompt`` accepts a client-supplied ``prompt_id`` verbatim and
two prompts can share an id. ``main.py``'s queue worker pins the active token
on the server around each ``e.execute(...)`` and the merge in ``send_sync``
reads that pinned token, so each prompt's events get its own metadata
regardless of ``prompt_id`` collisions or queue-vs-stack ordering.
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
    """``merge_prompt_metadata`` decorates execution events with the metadata
    for the currently-active token. Event payload fields win on conflict,
    binary payloads pass through, and the merge is gated on a ``prompt_id``
    marker to avoid decorating server-status events like ``status`` /
    ``queue_updated``."""

    def test_merges_for_active_token_when_payload_has_prompt_id(self, registry, lock):
        registry[42] = {"workflow_id": "wf-1"}
        merged = merge_prompt_metadata(registry, lock, 42, {"node": "n1", "prompt_id": "p1"})
        assert merged == {"node": "n1", "prompt_id": "p1", "workflow_id": "wf-1"}

    def test_passthrough_when_no_active_token(self, registry, lock):
        registry[42] = {"workflow_id": "wf-1"}
        merged = merge_prompt_metadata(registry, lock, None, {"node": "n1", "prompt_id": "p1"})
        assert merged == {"node": "n1", "prompt_id": "p1"}

    def test_passthrough_when_active_token_unknown(self, registry, lock):
        # Token was unregistered already (or never registered) — merge is a no-op.
        merged = merge_prompt_metadata(registry, lock, 99, {"prompt_id": "p1"})
        assert merged == {"prompt_id": "p1"}

    def test_passthrough_when_no_prompt_id(self, registry, lock):
        # Server-status frames (status, queue_updated, etc.) carry no prompt_id
        # and must not be decorated.
        registry[42] = {"workflow_id": "wf-1"}
        merged = merge_prompt_metadata(registry, lock, 42, {"status": {"queue_remaining": 0}})
        assert merged == {"status": {"queue_remaining": 0}}

    def test_passthrough_for_non_dict_payload(self, registry, lock):
        registry[42] = {"workflow_id": "wf-1"}
        binary = (b"image-bytes", {"prompt_id": "p1"})
        assert merge_prompt_metadata(registry, lock, 42, binary) is binary

    def test_event_payload_wins_over_registered_metadata(self, registry, lock):
        registry[42] = {"workflow_id": "wf-registered"}
        merged = merge_prompt_metadata(
            registry, lock, 42, {"prompt_id": "p1", "workflow_id": "wf-caller"}
        )
        assert merged["workflow_id"] == "wf-caller"


class TestProgressTextSidResolution:
    """``BinaryEventTypes.TEXT`` frames don't yet carry ``prompt_id`` /
    ``workflow_id`` in their wire shape, so cross-client routing has to happen
    at the ``sid`` level. The default sid pins the broadcast to the active
    prompt's client.
    """

    def test_explicit_sid_passes_through(self):
        assert resolve_progress_text_sid("client-explicit", "client-active") == "client-explicit"

    def test_none_sid_defaults_to_active_client(self):
        assert resolve_progress_text_sid(None, "client-active") == "client-active"

    def test_none_sid_with_no_active_client_stays_none(self):
        assert resolve_progress_text_sid(None, None) is None


class TestPromptIdCollisionWithTokens:
    """Two prompts can be queued with the same client-supplied ``prompt_id``.
    With a registry keyed by ``prompt_id`` the second registration would
    overwrite the first or be erased by the first's unregister. The token model
    makes each registration independent."""

    def test_two_submissions_get_distinct_tokens_and_each_merges_correctly(self, registry, lock):
        # Two submissions of the same prompt_id with different workflow_ids.
        registry[1] = {"workflow_id": "wf-A"}  # token from submission #1
        registry[2] = {"workflow_id": "wf-B"}  # token from submission #2

        # Worker is currently running submission #1.
        merged = merge_prompt_metadata(registry, lock, 1, {"prompt_id": "P", "node": "x"})
        assert merged["workflow_id"] == "wf-A"

        # Worker switches to submission #2 (queue ordering, retry, whatever).
        merged = merge_prompt_metadata(registry, lock, 2, {"prompt_id": "P", "node": "y"})
        assert merged["workflow_id"] == "wf-B"

    def test_unregister_by_token_does_not_drop_concurrent_submission(self, registry, lock):
        registry[1] = {"workflow_id": "wf-A"}
        registry[2] = {"workflow_id": "wf-B"}

        # Submission #1 finishes — drop its token only.
        registry.pop(1, None)

        # Submission #2 still has its metadata.
        merged = merge_prompt_metadata(registry, lock, 2, {"prompt_id": "P"})
        assert merged["workflow_id"] == "wf-B"

    def test_execution_order_independent_of_registration_order(self, registry, lock):
        """Regression for the LIFO-stack failure mode: queue executes #1 first
        but the previous stack design would have made the merge pick the
        latest-registered (#2) metadata. Token model is immune."""
        registry[1] = {"workflow_id": "wf-A"}
        registry[2] = {"workflow_id": "wf-B"}

        # Even though #2 was registered after #1, executing #1 still sees wf-A.
        merged_first_running = merge_prompt_metadata(registry, lock, 1, {"prompt_id": "P"})
        assert merged_first_running["workflow_id"] == "wf-A"


class TestRaceRegressionForTerminalExecutingFrame:
    """Regression for the PR #13684 finally-clear race.

    Executor's ``finally`` previously cleared the workflow_id source, so the
    post-completion terminal frame shipped ``workflow_id=None``. With the
    token model, the active token stays pinned until ``main.py`` clears it
    *after* the terminal send.
    """

    def test_terminal_executing_frame_includes_workflow_id(self, registry, lock):
        registry[7] = {"workflow_id": "wf-1"}
        active_token = 7

        # main.py emits the terminal frame BEFORE clearing the active token.
        terminal = merge_prompt_metadata(
            registry, lock, active_token, {"node": None, "prompt_id": "p1"}
        )
        # main.py's finally: clear active token + unregister.
        active_token = None
        registry.pop(7, None)

        assert terminal == {"node": None, "prompt_id": "p1", "workflow_id": "wf-1"}

        # After cleanup, straggler events get no metadata.
        straggler = merge_prompt_metadata(
            registry, lock, active_token, {"node": None, "prompt_id": "p1"}
        )
        assert "workflow_id" not in straggler
