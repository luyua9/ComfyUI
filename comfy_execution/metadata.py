"""Per-prompt metadata propagated alongside execution WebSocket events.

The execution layer (``execution.py``) is intentionally kept agnostic of
workflow-level concepts. ``PromptServer`` registers metadata at submission time
and merges it onto outgoing WebSocket payloads in ``send_sync``. Today only
``workflow_id`` is propagated; the structure is a ``TypedDict`` so additional
keys can be added without churn at the call sites.

Identity model — registry is keyed by an internal monotonic token, NOT by
``prompt_id``. ``post_prompt`` accepts a client-supplied ``prompt_id``
verbatim, so two prompts can be queued with the same id and a registry keyed
only by ``prompt_id`` would misattribute events when queue order differs from
registration order. ``main.py``'s queue worker pins the active token on the
server for the duration of one prompt's execution and ``send_sync`` reads that
token — so each prompt's events get its own registered metadata regardless of
``prompt_id`` collisions.
"""

import threading
from typing import Optional, TypedDict

from comfy_execution.jobs import extract_workflow_id


PROMPT_METADATA_TOKEN_KEY = "__prompt_metadata_token"


class PromptMetadata(TypedDict, total=False):
    workflow_id: Optional[str]


def build_prompt_metadata(extra_data: Optional[dict]) -> PromptMetadata:
    """Build a ``PromptMetadata`` snapshot from a prompt's ``extra_data``.

    Returns an empty dict when no recognized metadata is present so callers can
    skip registering anything in that case.
    """
    meta: PromptMetadata = {}
    workflow_id = extract_workflow_id(extra_data)
    if workflow_id is not None:
        meta["workflow_id"] = workflow_id
    return meta


def resolve_progress_text_sid(sid, default_sid):
    """Pick the recipient for a ``send_progress_text`` binary frame.

    Returns ``default_sid`` (typically ``PromptServer.client_id`` — the client
    that submitted the active prompt) when the caller didn't pin a specific
    socket. This narrows the audience for text status updates from "every
    connected client" to "the client running this prompt", matching the
    cross-client isolation other execution events already have.

    Splitting this out keeps the unit test independent of the full ``server``
    import chain.
    """
    return default_sid if sid is None else sid


def merge_prompt_metadata(
    registry: dict,
    lock: threading.Lock,
    active_token: Optional[int],
    data,
):
    """Return ``data`` with the metadata for the currently-active token merged
    top-level. The event payload wins on conflict, and non-dict payloads (e.g.
    the binary preview tuple) pass through untouched.

    The active token is set by ``main.py``'s queue worker around each
    ``e.execute(...)``. The merge happens only when a token is currently active
    *and* the payload carries a ``prompt_id`` — using ``prompt_id`` purely as a
    marker that this is an execution event meant to receive metadata, not as
    the lookup key. This makes the merge immune to ``prompt_id`` collisions.
    """
    if not isinstance(data, dict):
        return data
    if not data.get("prompt_id"):
        return data
    if active_token is None:
        return data
    with lock:
        meta = registry.get(active_token)
    if not meta:
        return data
    return {**meta, **data}
