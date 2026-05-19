"""Per-prompt metadata propagated alongside execution WebSocket events.

The execution layer (``execution.py``) is intentionally kept agnostic of
workflow-level concepts. Instead, ``PromptServer`` registers per-``prompt_id``
metadata at submission time and merges it onto outgoing WebSocket payloads in
``send_sync``. Today only ``workflow_id`` is propagated; the structure is a
``TypedDict`` so additional keys can be added without churn at the call sites.
"""

import threading
from typing import Optional, TypedDict

from comfy_execution.jobs import extract_workflow_id


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
    data,
):
    """Return ``data`` with the registered metadata for its ``prompt_id`` merged
    top-level. The event payload wins on conflict, and non-dict payloads (e.g.
    the binary preview tuple) pass through untouched.

    The registry is a stack per ``prompt_id`` (``dict[str, list[PromptMetadata]]``)
    so duplicate submissions of the same ``prompt_id`` don't clobber each
    other's metadata; the most recently registered entry wins.
    """
    if not isinstance(data, dict):
        return data
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        return data
    with lock:
        stack = registry.get(prompt_id)
        meta = stack[-1] if stack else None
    if not meta:
        return data
    return {**meta, **data}
