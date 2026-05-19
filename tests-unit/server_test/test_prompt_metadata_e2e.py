"""End-to-end checks of the prompt-metadata propagation.

These tests instantiate the real ``PromptServer`` (no heavy mocking) and drive
the same call sequence ``main.py`` would: register metadata, pin the token,
emit a bunch of execution-shaped frames through ``send_sync``, then clear the
token + unregister. We then drain the server's message queue and inspect the
exact frames that would have been written to the WebSocket — proving the wire
shape and the ``prompt_id`` collision invariants without needing a browser.
"""

import asyncio
import os
import sys
import threading

import pytest

# Ensure the repo root is the first thing on sys.path so ``utils`` resolves to
# the ComfyUI package, not a site-packages namespace package. ``utils`` must
# load before ``app.frontend_management`` imports it.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import utils  # noqa: F401  -- pin the package before later imports

from comfy_execution.metadata import PROMPT_METADATA_TOKEN_KEY

server_module = pytest.importorskip("server")


@pytest.fixture
def prompt_server():
    loop = asyncio.new_event_loop()
    try:
        server = server_module.PromptServer(loop)
        yield server
    finally:
        loop.close()


def _drain(server) -> list:
    """Empty the server's internal asyncio.Queue and return every (event, data, sid)."""
    frames = []
    queue = server.messages
    while not queue.empty():
        frames.append(queue.get_nowait())
    return frames


def _run_pending_callbacks(server):
    """``send_sync`` schedules ``messages.put_nowait`` via ``loop.call_soon_threadsafe``.
    Pump the loop a single iteration so those callbacks land before we drain."""
    server.loop.call_soon(server.loop.stop)
    server.loop.run_forever()


def _simulate_prompt_lifecycle(server, extra_data: dict, prompt_id: str):
    """Replicates what ``main.py``'s queue worker does around ``e.execute(...)``
    plus a representative sample of the events ``execution.py`` emits."""
    token = server.register_prompt_metadata(extra_data)
    if token is not None:
        extra_data[PROMPT_METADATA_TOKEN_KEY] = token

    # Worker picks up the item, pins the token.
    server.active_prompt_metadata_token = extra_data.pop(PROMPT_METADATA_TOKEN_KEY, None)
    client_id = "client-A"
    server.client_id = client_id

    # A representative sample of the eight execution events listed in PR #13684.
    server.send_sync("execution_start", {"prompt_id": prompt_id}, client_id)
    server.send_sync("execution_cached", {"nodes": [], "prompt_id": prompt_id}, client_id)
    server.send_sync("executing", {"node": "n1", "display_node": "n1", "prompt_id": prompt_id}, client_id)
    server.send_sync(
        "executed",
        {"node": "n1", "display_node": "n1", "output": {}, "prompt_id": prompt_id},
        client_id,
    )
    server.send_sync("execution_success", {"prompt_id": prompt_id}, client_id)
    # The terminal frame ``main.py`` itself emits, just before clearing state.
    server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, client_id)

    # Worker finally clause: clear token then unregister.
    server.active_prompt_metadata_token = None
    server.unregister_prompt_metadata(token)


class TestEndToEndWorkflowIdOnAllExecutionEvents:
    def test_every_execution_event_carries_workflow_id(self, prompt_server):
        _simulate_prompt_lifecycle(
            prompt_server,
            extra_data={"extra_pnginfo": {"workflow": {"id": "wf-xyz"}}},
            prompt_id="p-1",
        )
        _run_pending_callbacks(prompt_server)
        frames = _drain(prompt_server)

        assert frames, "no frames emitted"
        # Every frame with a prompt_id payload must carry workflow_id top-level.
        for event, data, _sid in frames:
            if isinstance(data, dict) and data.get("prompt_id"):
                assert data.get("workflow_id") == "wf-xyz", (event, data)

        # And specifically verify the terminal frame — the #13684 race victim.
        terminal = [
            (e, d) for e, d, _ in frames
            if e == "executing" and isinstance(d, dict) and d.get("node") is None
        ]
        assert terminal, "no terminal executing frame emitted"
        _, terminal_payload = terminal[-1]
        assert terminal_payload["workflow_id"] == "wf-xyz"

    def test_status_frame_is_not_decorated(self, prompt_server):
        prompt_server.register_prompt_metadata({"extra_pnginfo": {"workflow": {"id": "wf-1"}}})
        prompt_server.active_prompt_metadata_token = 1

        # A status frame has no prompt_id and must remain untouched.
        prompt_server.send_sync("status", {"status": {"queue_remaining": 0}}, None)
        _run_pending_callbacks(prompt_server)
        frames = _drain(prompt_server)

        assert any(e == "status" for e, _, _ in frames)
        for event, data, _ in frames:
            if event == "status":
                assert "workflow_id" not in data


class TestEndToEndPromptIdCollision:
    """Drive two prompts with the same client-supplied ``prompt_id`` but
    different ``workflow_id`` values. The token model must keep their event
    streams attributed correctly."""

    def test_same_prompt_id_two_workflows_each_stream_attributed_correctly(self, prompt_server):
        _simulate_prompt_lifecycle(
            prompt_server,
            extra_data={"extra_pnginfo": {"workflow": {"id": "wf-A"}}},
            prompt_id="P-shared",
        )
        _simulate_prompt_lifecycle(
            prompt_server,
            extra_data={"extra_pnginfo": {"workflow": {"id": "wf-B"}}},
            prompt_id="P-shared",
        )
        _run_pending_callbacks(prompt_server)
        frames = _drain(prompt_server)

        # Partition the frames by lifecycle (execution_start..terminal-executing).
        runs: list[list[dict]] = [[]]
        for event, data, _ in frames:
            if not isinstance(data, dict):
                continue
            runs[-1].append({"event": event, **data})
            if event == "executing" and data.get("node") is None:
                runs.append([])
        # The trailing empty bucket from the final split is fine.
        runs = [r for r in runs if r]
        assert len(runs) == 2, runs

        run_a, run_b = runs
        for entry in run_a:
            if entry.get("prompt_id"):
                assert entry.get("workflow_id") == "wf-A", entry
        for entry in run_b:
            if entry.get("prompt_id"):
                assert entry.get("workflow_id") == "wf-B", entry
