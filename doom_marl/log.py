"""Shared :mod:`rich`-backed logging helpers used across the project.

All console output should go through :func:`log_event`, :func:`log_info`,
:func:`log_warn`, :func:`log_success` and friends rather than bare ``print``
calls. The module owns a single :class:`rich.console.Console` (writing to
``stderr`` so it never interleaves with stdout pipes) and renders messages in a
consistent ``HH:MM:SS [tag] status key=value...`` format.

Notes on multi-process usage: workers spawned by ``torchrl``'s ParallelEnv
inherit no state from the main process, so each worker constructs its own
``Console``. That is fine — every process just writes its own styled lines to
its own stderr; no shared state is needed.
"""
from __future__ import annotations

import io
import os
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class _TqdmStream(io.TextIOBase):
    """File-like wrapper that routes writes through ``tqdm.write`` when a
    tqdm progress bar is active, so log lines don't shred the bar.

    Falls back to writing directly to the underlying stream when tqdm isn't
    importable or no bar is live (e.g. in worker subprocesses).
    """

    def __init__(self, stream):
        self._stream = stream

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def isatty(self) -> bool:  # type: ignore[override]
        return getattr(self._stream, "isatty", lambda: False)()

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        try:
            from tqdm import tqdm  # local import to avoid hard dep at import
            if getattr(tqdm, "_instances", None):
                # ``tqdm.write`` always appends a newline; strip a single
                # trailing one from ``s`` to avoid doubling it (rich already
                # ends lines with "\n").
                msg = s[:-1] if s.endswith("\n") else s
                tqdm.write(msg, file=self._stream, end="\n" if s.endswith("\n") else "")
                return len(s)
        except Exception:
            pass
        self._stream.write(s)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._stream.flush()
        except Exception:
            pass


# Single shared console. We write through a tqdm-aware stderr wrapper so
# log lines and the BenchMARL progress bar don't clobber each other.
# Disabling ``highlight`` prevents rich from reformatting numbers/paths,
# keeping key=value pairs visually consistent.
_CONSOLE: Console = Console(
    file=_TqdmStream(sys.stderr), highlight=False, soft_wrap=True
)


def get_console() -> Console:
    """Return the shared :class:`Console` used for project-wide logging."""
    return _CONSOLE


# --------------------------------------------------------------------------
# Severity styles
# --------------------------------------------------------------------------

_LEVEL_STYLES: dict[str, str] = {
    "info":     "bold cyan",
    "warn":     "bold yellow",
    "error":    "bold red",
    "success":  "bold green",
    "debug":    "dim",
}

_STATUS_STYLES: dict[str, str] = {
    "starting":  "yellow",
    "ready":     "bold green",
    "failed":    "bold red",
    "saved":     "green",
    "uploaded":  "cyan",
    "enabled":   "bold green",
    "disabled":  "dim yellow",
    "skipped":   "dim",
    "warning":   "bold yellow",
}


def _format_fields(fields: dict[str, Any]) -> Text:
    """Render ``key=value`` pairs as ``key=[bold]value[/]`` with a dim ``key=``."""
    text = Text()
    first = True
    for key, value in fields.items():
        if value is None:
            continue
        if not first:
            text.append(" ")
        first = False
        text.append(f"{key}=", style="dim")
        text.append(str(value))
    return text


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def log_event(
    tag: str,
    status: str | None = None,
    *,
    level: str = "info",
    pid: bool = False,
    **fields: Any,
) -> None:
    """Log a structured event line.

    Format::
        [tag] status key1=v1 key2=v2 ...

    ``tag`` is shown in the level color, ``status`` (if given) in a status
    color (``starting``/``ready``/``failed``/...). Remaining keyword args are
    appended as dim ``key=value`` pairs.
    """
    msg = Text()
    msg.append(f"[{tag}] ", style=_LEVEL_STYLES.get(level, "bold cyan"))
    if status is not None:
        style = _STATUS_STYLES.get(status, "bold")
        msg.append(status, style=style)
        if fields or pid:
            msg.append(" ")
    if pid:
        msg.append("pid=", style="dim")
        msg.append(str(os.getpid()))
        if fields:
            msg.append(" ")
    if fields:
        msg.append_text(_format_fields(fields))
    _CONSOLE.print(msg)


def log_info(message: str, *, tag: str | None = None) -> None:
    _emit_message(message, level="info", tag=tag)


def log_warn(message: str, *, tag: str | None = None) -> None:
    _emit_message(message, level="warn", tag=tag)


def log_error(message: str, *, tag: str | None = None) -> None:
    _emit_message(message, level="error", tag=tag)


def log_success(message: str, *, tag: str | None = None) -> None:
    _emit_message(message, level="success", tag=tag)


def _emit_message(message: str, *, level: str, tag: str | None) -> None:
    out = Text()
    if tag is not None:
        out.append(f"[{tag}] ", style=_LEVEL_STYLES.get(level, "bold cyan"))
    out.append(message, style=_LEVEL_STYLES.get(level, ""))
    _CONSOLE.print(out)


def log_panel(title: str, body: Any, *, style: str = "cyan") -> None:
    """Print ``body`` (a Rich renderable or string) inside a titled panel."""
    _CONSOLE.print(Panel(body, title=title, border_style=style, padding=(0, 1)))


def log_kv_block(title: str, items: dict[str, Any], *, style: str = "cyan") -> None:
    """Print a small key/value block under a header line.

    Used for short summaries (devices, workspace urls) where a full table is
    overkill but plain prints look ugly.
    """
    body = Text()
    width = max((len(k) for k in items), default=0)
    for i, (k, v) in enumerate(items.items()):
        if i > 0:
            body.append("\n")
        body.append(f"{k:<{width}} ", style="dim")
        body.append(str(v))
    log_panel(title, body, style=style)
